/* _gpythonmodule.c -- the CPython extension over gkyl_gpython.h
 * (GKEYLL_C_SHIM.md).
 *
 * Knows Python objects, NumPy arrays, and the gpython contract -- and nothing
 * else about Gkeyll: gkyl_gpython.h (the gpython shim, which lives in the
 * gkeyll repo and is compiled into libg0core.so) exposes only opaque handles,
 * scalars, and buffers, so no layout or calling convention exists on this
 * side of the wall.
 *
 * Ownership model: native handles live in PyCapsules whose destructors
 * release the C object; a capsule's context slot optionally pins a Python
 * buffer the C array aliases (zero-copy construction). NumPy views of
 * array data take the capsule as their base, so a view keeps the native
 * memory alive for as long as the view itself is reachable.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/* Pin the NumPy ABI this extension targets to postgkyl's own declared floor
 * (numpy>=2.2.6 in pyproject.toml). Without this, NumPy's headers default
 * PyArray_Descr et al. to whatever the *compiling* NumPy's minor version
 * happens to be, so a `_gpython.so` built against e.g. NumPy 2.5 headers can
 * fail NumPy's own import_array() ABI check ("numpy.dtype size changed")
 * against a NumPy 2.2 runtime -- a real hazard here because pip's isolated
 * build environment resolves build-system.requires' numpy independently
 * from the numpy actually installed into the target environment.
 *
 * This is a best-effort backstop, NOT a substitute for building against the
 * actual runtime NumPy: a build/runtime version skew has been reproduced to
 * crash outright (segfault / heap corruption inside Gkeyll's C code, not a
 * clean import_array() failure) even with this pin in place -- see
 * scripts/build_gpython.sh and README.md's "--no-build-isolation" install
 * instructions, which are the real fix.
 */
#define NPY_TARGET_VERSION NPY_2_2_API_VERSION
#include <numpy/arrayobject.h>

#include <errno.h>

#include <gkyl_gpython.h>

static const char ARRAY_CAP[] = "gpython_array";
static const char BASIS_CAP[] = "gpython_basis";

/* ------------------------------------------------------------ capsules */
static void
array_capsule_destroy(PyObject *cap)
{
  gpython_array *a = PyCapsule_GetPointer(cap, ARRAY_CAP);
  if (a)
    gpython_array_release(a);
  Py_XDECREF((PyObject *)PyCapsule_GetContext(cap));
}

static void
basis_capsule_destroy(PyObject *cap)
{
  gpython_basis *b = PyCapsule_GetPointer(cap, BASIS_CAP);
  if (b)
    gpython_basis_release(b);
}

static gpython_array *
array_arg(PyObject *cap)
{
  return (gpython_array *)PyCapsule_GetPointer(cap, ARRAY_CAP);
}

static gpython_basis *
basis_arg(PyObject *cap)
{
  return (gpython_basis *)PyCapsule_GetPointer(cap, BASIS_CAP);
}

static PyObject *
wrap_array(gpython_array *a)
{
  if (!a) {
    PyErr_SetString(PyExc_MemoryError, "received NULL gpython_array");
    return NULL;
  }
  PyObject *cap = PyCapsule_New(a, ARRAY_CAP, array_capsule_destroy);
  if (!cap)
    gpython_array_release(a);
  return cap;
}

/* --------------------------------------------------------------- misc */
static PyObject *
py_api_version(PyObject *self, PyObject *noargs)
{
  return PyLong_FromLong(gpython_api_version());
}

/* -------------------------------------------------------------- arrays */
static PyObject *
py_array_new(PyObject *self, PyObject *args)
{
  Py_ssize_t ncomp, size;
  if (!PyArg_ParseTuple(args, "nn", &ncomp, &size))
    return NULL;
  return wrap_array(gpython_array_new((size_t)ncomp, (size_t)size));
}

static PyObject *
py_array_from_numpy(PyObject *self, PyObject *args)
{
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;
  /* A C-contiguous float64 array; copies only if the input is not already
   * one. The result is pinned in the capsule context: the C array is a
   * zero-copy view of exactly this buffer. */
  PyArrayObject *buf =
      (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!buf)
    return NULL;
  if (PyArray_NDIM(buf) < 1) {
    Py_DECREF(buf);
    PyErr_SetString(PyExc_ValueError, "need at least a 1-D (…, ncomp) array");
    return NULL;
  }
  npy_intp ncomp = PyArray_DIM(buf, PyArray_NDIM(buf) - 1);
  npy_intp size = PyArray_SIZE(buf) / (ncomp ? ncomp : 1);
  gpython_array *a =
      gpython_array_from_buff((size_t)ncomp, (size_t)size, PyArray_DATA(buf));
  PyObject *cap = wrap_array(a);
  if (!cap) {
    Py_DECREF(buf);
    return NULL;
  }
  PyCapsule_SetContext(cap, buf); /* pin: released by the capsule dtor */
  return cap;
}

static PyObject *
py_array_clone(PyObject *self, PyObject *args)
{
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  gpython_array *a = array_arg(cap);
  if (!a)
    return NULL;
  return wrap_array(gpython_array_clone(a));
}

static PyObject *
py_array_ncomp(PyObject *self, PyObject *args)
{
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  gpython_array *a = array_arg(cap);
  if (!a)
    return NULL;
  return PyLong_FromSize_t(gpython_array_ncomp(a));
}

static PyObject *
py_array_size(PyObject *self, PyObject *args)
{
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  gpython_array *a = array_arg(cap);
  if (!a)
    return NULL;
  return PyLong_FromSize_t(gpython_array_size(a));
}

static PyObject *
py_array_view(PyObject *self, PyObject *args)
{
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  gpython_array *a = array_arg(cap);
  if (!a)
    return NULL;
  npy_intp dims[2] = {(npy_intp)gpython_array_size(a),
                      (npy_intp)gpython_array_ncomp(a)};
  PyObject *view =
      PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, gpython_array_data(a));
  if (!view)
    return NULL;
  Py_INCREF(cap); /* base steals this reference */
  if (PyArray_SetBaseObject((PyArrayObject *)view, cap) < 0) {
    Py_DECREF(view);
    return NULL;
  }
  PyArray_CLEARFLAGS((PyArrayObject *)view, NPY_ARRAY_WRITEABLE);
  return view;
}

/* ------------------------------------------------------------- file I/O */
static PyObject *
grid_tuple(int ndim, const double *lower, const double *upper, const int *cells)
{
  npy_intp n = ndim;
  PyObject *lo = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
  PyObject *up = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
  PyObject *nc = PyArray_SimpleNew(1, &n, NPY_INT64);
  if (!lo || !up || !nc) {
    Py_XDECREF(lo);
    Py_XDECREF(up);
    Py_XDECREF(nc);
    return NULL;
  }
  for (int d = 0; d < ndim; ++d) {
    ((double *)PyArray_DATA((PyArrayObject *)lo))[d] = lower[d];
    ((double *)PyArray_DATA((PyArrayObject *)up))[d] = upper[d];
    ((npy_int64 *)PyArray_DATA((PyArrayObject *)nc))[d] = cells[d];
  }
  return Py_BuildValue("(iNNN)", ndim, lo, up, nc);
}

static PyObject *
py_file_type(PyObject *self, PyObject *args)
{
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname))
    return NULL;
  return PyLong_FromLong(gpython_file_type(fname));
}

static PyObject *
py_read_header(PyObject *self, PyObject *args)
{
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname))
    return NULL;
  int ndim, file_type, cells[gpython_MAX_DIM];
  double lower[gpython_MAX_DIM], upper[gpython_MAX_DIM];
  size_t esznc, tot_cells, meta_sz;
  char *meta;
  int status =
      gpython_read_header(fname, &ndim, lower, upper, cells, &file_type, &esznc,
                          &tot_cells, &meta, &meta_sz);
  if (status != 0) {
    PyErr_Format(PyExc_OSError, "'%s': %s", fname, gpython_status_msg(status));
    return NULL;
  }
  PyObject *grid = grid_tuple(ndim, lower, upper, cells);
  PyObject *meta_bytes =
      PyBytes_FromStringAndSize(meta ? meta : "", (Py_ssize_t)meta_sz);
  gpython_meta_release(meta);
  if (!grid || !meta_bytes) {
    Py_XDECREF(grid);
    Py_XDECREF(meta_bytes);
    return NULL;
  }
  return Py_BuildValue("(NiNnn)", grid, file_type, meta_bytes,
                       (Py_ssize_t)esznc, (Py_ssize_t)tot_cells);
}

static PyObject *
py_read_field(PyObject *self, PyObject *args)
{
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname))
    return NULL;
  int ndim, cells[gpython_MAX_DIM];
  double lower[gpython_MAX_DIM], upper[gpython_MAX_DIM];
  gpython_array *a = gpython_read_field(fname, &ndim, lower, upper, cells);
  if (!a) {
    PyErr_Format(PyExc_OSError, "'%s': gpython_read_field failed", fname);
    return NULL;
  }
  PyObject *grid = grid_tuple(ndim, lower, upper, cells);
  PyObject *cap = wrap_array(a);
  if (!grid || !cap) {
    Py_XDECREF(grid);
    Py_XDECREF(cap);
    return NULL;
  }
  return Py_BuildValue("(NN)", grid, cap);
}

/* --------------------------------------------------------------- basis */
static PyObject *
py_basis_new(PyObject *self, PyObject *args)
{
  const char *type;
  int ndim, poly_order;
  if (!PyArg_ParseTuple(args, "sii", &type, &ndim, &poly_order))
    return NULL;
  gpython_basis *b = gpython_basis_new(type, ndim, poly_order);
  if (!b) {
    PyErr_Format(PyExc_NotImplementedError,
                 "basis '%s' is not wired through the Gkeyll shim", type);
    return NULL;
  }
  return PyCapsule_New(b, BASIS_CAP, basis_capsule_destroy);
}

static PyObject *
py_basis_new_hybrid(PyObject *self, PyObject *args)
{
  const char *type;
  int cdim, vdim;
  if (!PyArg_ParseTuple(args, "sii", &type, &cdim, &vdim))
    return NULL;
  gpython_basis *b = gpython_basis_new_hybrid(type, cdim, vdim);
  if (!b) {
    PyErr_Format(PyExc_NotImplementedError,
                 "basis '%s' is not wired through the Gkeyll shim", type);
    return NULL;
  }
  return PyCapsule_New(b, BASIS_CAP, basis_capsule_destroy);
}

static PyObject *
py_basis_info(PyObject *self, PyObject *args)
{
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  gpython_basis *b = basis_arg(cap);
  if (!b)
    return NULL;
  return Py_BuildValue("(iiis)", gpython_basis_ndim(b),
                       gpython_basis_poly_order(b), gpython_basis_num_basis(b),
                       gpython_basis_id(b));
}

static PyObject *
py_basis_eval(PyObject *self, PyObject *args)
{
  PyObject *cap, *zobj;
  if (!PyArg_ParseTuple(args, "OO", &cap, &zobj))
    return NULL;
  gpython_basis *b = basis_arg(cap);
  if (!b)
    return NULL;
  PyArrayObject *z =
      (PyArrayObject *)PyArray_FROM_OTF(zobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!z)
    return NULL;
  if (PyArray_SIZE(z) < gpython_basis_ndim(b)) {
    Py_DECREF(z);
    PyErr_SetString(PyExc_ValueError, "point has fewer entries than ndim");
    return NULL;
  }
  npy_intp nb = gpython_basis_num_basis(b);
  PyObject *out = PyArray_SimpleNew(1, &nb, NPY_DOUBLE);
  if (!out) {
    Py_DECREF(z);
    return NULL;
  }
  gpython_basis_eval(b, PyArray_DATA(z), PyArray_DATA((PyArrayObject *)out));
  Py_DECREF(z);
  return out;
}

static PyObject *
py_basis_node_list(PyObject *self, PyObject *args)
{
  PyObject *cap;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  gpython_basis *b = basis_arg(cap);
  if (!b)
    return NULL;
  npy_intp dims[2] = {gpython_basis_num_basis(b), gpython_basis_ndim(b)};
  PyObject *out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (!out)
    return NULL;
  gpython_basis_node_list(b, PyArray_DATA((PyArrayObject *)out));
  return out;
}

static PyObject *
py_basis_nodal_to_modal(PyObject *self, PyObject *args)
{
  PyObject *cap, *fobj;
  if (!PyArg_ParseTuple(args, "OO", &cap, &fobj))
    return NULL;
  gpython_basis *b = basis_arg(cap);
  if (!b)
    return NULL;
  PyArrayObject *fin =
      (PyArrayObject *)PyArray_FROM_OTF(fobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!fin)
    return NULL;
  npy_intp nb = gpython_basis_num_basis(b);
  if (PyArray_SIZE(fin) != nb) {
    Py_DECREF(fin);
    PyErr_SetString(PyExc_ValueError, "expected num_basis nodal values");
    return NULL;
  }
  PyObject *out = PyArray_SimpleNew(1, &nb, NPY_DOUBLE);
  if (!out) {
    Py_DECREF(fin);
    return NULL;
  }
  gpython_basis_nodal_to_modal(b, PyArray_DATA(fin),
                               PyArray_DATA((PyArrayObject *)out));
  Py_DECREF(fin);
  return out;
}

/* ------------------------------------------------------ weak DG algebra */
typedef int (*binop_fn)(const gpython_basis *, gpython_array *,
                        const gpython_array *, const gpython_array *);

static PyObject *
binop(PyObject *args, binop_fn fn, const char *name)
{
  PyObject *bcap, *ocap, *acap, *bcap2;
  if (!PyArg_ParseTuple(args, "OOOO", &bcap, &ocap, &acap, &bcap2))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *out = array_arg(ocap), *a1 = array_arg(acap),
                *a2 = array_arg(bcap2);
  if (!b || !out || !a1 || !a2)
    return NULL;
  if (fn(b, out, a1, a2) != 0) {
    PyErr_Format(PyExc_ValueError,
                 "%s: operand shapes incompatible with "
                 "the basis (ncomp must be a multiple of num_basis and match)",
                 name);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *
py_dg_mul(PyObject *self, PyObject *args)
{
  return binop(args, gpython_dg_mul, "dg_mul");
}

static PyObject *
py_dg_div(PyObject *self, PyObject *args)
{
  return binop(args, gpython_dg_div, "dg_div");
}

static PyObject *
py_dg_inv(PyObject *self, PyObject *args)
{
  PyObject *bcap, *ocap, *acap;
  if (!PyArg_ParseTuple(args, "OOO", &bcap, &ocap, &acap))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *out = array_arg(ocap), *a1 = array_arg(acap);
  if (!b || !out || !a1)
    return NULL;
  if (gpython_dg_inv(b, out, a1) != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "dg_inv: operand shapes incompatible with the basis");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *
py_dg_mul_conf_phase(PyObject *self, PyObject *args)
{
  PyObject *cbcap, *pbcap, *poutcap, *copcap, *popcap, *ccellsobj, *pcellsobj;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &cbcap, &pbcap, &poutcap, &copcap,
                        &popcap, &ccellsobj, &pcellsobj))
    return NULL;
  gpython_basis *cbasis = basis_arg(cbcap), *pbasis = basis_arg(pbcap);
  gpython_array *pout = array_arg(poutcap), *cop = array_arg(copcap),
                *pop = array_arg(popcap);
  if (!cbasis || !pbasis || !pout || !cop || !pop)
    return NULL;
  PyArrayObject *ccells = (PyArrayObject *)PyArray_FROM_OTF(
      ccellsobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *pcells = (PyArrayObject *)PyArray_FROM_OTF(
      pcellsobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (!ccells || !pcells) {
    Py_XDECREF(ccells);
    Py_XDECREF(pcells);
    return NULL;
  }
  if (PyArray_SIZE(ccells) != gpython_basis_ndim(cbasis) ||
      PyArray_SIZE(pcells) != gpython_basis_ndim(pbasis)) {
    Py_DECREF(ccells);
    Py_DECREF(pcells);
    PyErr_SetString(
        PyExc_ValueError,
        "mul_conf_phase: cells arrays must match each basis's ndim");
    return NULL;
  }
  int status =
      gpython_dg_mul_conf_phase(cbasis, pbasis, pout, cop, pop,
                                PyArray_DATA(ccells), PyArray_DATA(pcells));
  Py_DECREF(ccells);
  Py_DECREF(pcells);
  if (status != 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "mul_conf_phase: operand shapes incompatible with the bases/cells");
    return NULL;
  }
  Py_RETURN_NONE;
}

/* --------------------------------------- linear coefficient ops / reduce */
static PyObject *
py_array_set(PyObject *self, PyObject *args)
{
  PyObject *ocap, *acap;
  double c;
  if (!PyArg_ParseTuple(args, "OdO", &ocap, &c, &acap))
    return NULL;
  gpython_array *out = array_arg(ocap), *a = array_arg(acap);
  if (!out || !a)
    return NULL;
  gpython_array_set(out, c, a);
  Py_RETURN_NONE;
}

static PyObject *
py_array_accumulate(PyObject *self, PyObject *args)
{
  PyObject *ocap, *acap;
  double c;
  if (!PyArg_ParseTuple(args, "OdO", &ocap, &c, &acap))
    return NULL;
  gpython_array *out = array_arg(ocap), *a = array_arg(acap);
  if (!out || !a)
    return NULL;
  gpython_array_accumulate(out, c, a);
  Py_RETURN_NONE;
}

static PyObject *
py_array_scale(PyObject *self, PyObject *args)
{
  PyObject *acap;
  double c;
  if (!PyArg_ParseTuple(args, "Od", &acap, &c))
    return NULL;
  gpython_array *a = array_arg(acap);
  if (!a)
    return NULL;
  gpython_array_scale(a, c);
  Py_RETURN_NONE;
}

static PyObject *
py_array_shiftc(PyObject *self, PyObject *args)
{
  PyObject *acap;
  double val;
  unsigned comp;
  if (!PyArg_ParseTuple(args, "OdI", &acap, &val, &comp))
    return NULL;
  gpython_array *a = array_arg(acap);
  if (!a)
    return NULL;
  gpython_array_shiftc(a, val, comp);
  Py_RETURN_NONE;
}

static PyObject *
py_array_reduce(PyObject *self, PyObject *args)
{
  PyObject *acap;
  int op;
  if (!PyArg_ParseTuple(args, "Oi", &acap, &op))
    return NULL;
  gpython_array *a = array_arg(acap);
  if (!a)
    return NULL;
  if (op < 0 || op > 2) {
    PyErr_SetString(PyExc_ValueError, "reduce op must be 0/1/2 (min/max/sum)");
    return NULL;
  }
  npy_intp n = (npy_intp)gpython_array_ncomp(a);
  PyObject *out = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
  if (!out)
    return NULL;
  gpython_array_reduce(PyArray_DATA((PyArrayObject *)out), a, op);
  return out;
}

/* ---------------------------------------------------- field-aware reduce */
static PyObject *
py_array_dg_reduce(PyObject *self, PyObject *args)
{
  PyObject *bcap, *acap;
  int comp, op;
  if (!PyArg_ParseTuple(args, "OOii", &bcap, &acap, &comp, &op))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *a = array_arg(acap);
  if (!b || !a)
    return NULL;
  if (op < 0 || op > 2) {
    PyErr_SetString(PyExc_ValueError, "reduce op must be 0/1/2 (min/max/sum)");
    return NULL;
  }
  double out;
  if (gpython_array_dg_reduce(&out, b, a, comp, op) != 0) {
    PyErr_Format(PyExc_ValueError,
                 "dg_reduce: component %d out of range for this basis", comp);
    return NULL;
  }
  return PyFloat_FromDouble(out);
}

/* ------------------------------------------------------------ integrate */
static PyObject *
py_array_integrate(PyObject *self, PyObject *args)
{
  PyObject *loobj, *upobj, *ncobj, *bcap, *acap;
  int nfields, op;
  double factor;
  if (!PyArg_ParseTuple(args, "OOOOiidO", &loobj, &upobj, &ncobj, &bcap,
                        &nfields, &op, &factor, &acap))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *a = array_arg(acap);
  if (!b || !a)
    return NULL;
  if (op < 0 || op > 2) {
    PyErr_SetString(PyExc_ValueError,
                    "integrate op must be 0/1/2 (none/abs/sq)");
    return NULL;
  }
  PyArrayObject *lo =
      (PyArrayObject *)PyArray_FROM_OTF(loobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *up =
      (PyArrayObject *)PyArray_FROM_OTF(upobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *nc =
      (PyArrayObject *)PyArray_FROM_OTF(ncobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (!lo || !up || !nc)
    goto fail;
  int ndim = (int)PyArray_SIZE(lo);
  if (ndim < 1 || ndim > gpython_MAX_DIM || PyArray_SIZE(up) != ndim ||
      PyArray_SIZE(nc) != ndim) {
    PyErr_SetString(PyExc_ValueError, "grid arrays must share ndim <= 7");
    goto fail;
  }
  npy_intp n = nfields;
  PyObject *out = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
  if (!out)
    goto fail;
  int status = gpython_array_integrate(ndim, PyArray_DATA(lo), PyArray_DATA(up),
                                       PyArray_DATA(nc), b, nfields, op, factor,
                                       a, PyArray_DATA((PyArrayObject *)out));
  Py_DECREF(lo);
  Py_DECREF(up);
  Py_DECREF(nc);
  if (status != 0) {
    Py_DECREF(out);
    PyErr_SetString(PyExc_ValueError,
                    "integrate: grid cells do not cover the array");
    return NULL;
  }
  return out;
fail:
  Py_XDECREF(lo);
  Py_XDECREF(up);
  Py_XDECREF(nc);
  return NULL;
}

/* ------------------------------------------------------------- average */
static PyObject *
py_array_average(PyObject *self, PyObject *args)
{
  PyObject *loobj, *upobj, *ncobj, *bcap, *bavgcap, *ncavgobj, *avgdimobj,
      *wcap, *acap, *ocap;
  if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &loobj, &upobj, &ncobj, &bcap,
                        &bavgcap, &ncavgobj, &avgdimobj, &wcap, &acap, &ocap))
    return NULL;
  gpython_basis *b = basis_arg(bcap), *b_avg = basis_arg(bavgcap);
  gpython_array *a = array_arg(acap), *out = array_arg(ocap);
  if (!b || !b_avg || !a || !out)
    return NULL;
  gpython_array *weight = NULL;
  if (wcap != Py_None) {
    weight = array_arg(wcap);
    if (!weight)
      return NULL;
  }
  PyArrayObject *lo =
      (PyArrayObject *)PyArray_FROM_OTF(loobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *up =
      (PyArrayObject *)PyArray_FROM_OTF(upobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *nc =
      (PyArrayObject *)PyArray_FROM_OTF(ncobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *ncavg = (PyArrayObject *)PyArray_FROM_OTF(ncavgobj, NPY_INT32,
                                                           NPY_ARRAY_IN_ARRAY);
  PyArrayObject *avgdim = (PyArrayObject *)PyArray_FROM_OTF(
      avgdimobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (!lo || !up || !nc || !ncavg || !avgdim)
    goto fail;
  int ndim = (int)PyArray_SIZE(lo);
  int ndim_avg = (int)PyArray_SIZE(ncavg);
  if (ndim < 1 || ndim > gpython_MAX_DIM || PyArray_SIZE(up) != ndim ||
      PyArray_SIZE(nc) != ndim || PyArray_SIZE(avgdim) != ndim) {
    PyErr_SetString(PyExc_ValueError,
                    "average: grid/avg_dim arrays must share ndim <= 7");
    goto fail;
  }
  if (ndim_avg < 1 || ndim_avg > gpython_MAX_DIM) {
    PyErr_SetString(PyExc_ValueError,
                    "average: cells_avg must have between 1 and 7 entries");
    goto fail;
  }
  int status = gpython_array_average(
      ndim, PyArray_DATA(lo), PyArray_DATA(up), PyArray_DATA(nc), b, b_avg,
      ndim_avg, PyArray_DATA(ncavg), PyArray_DATA(avgdim), weight, a, out);
  Py_DECREF(lo);
  Py_DECREF(up);
  Py_DECREF(nc);
  Py_DECREF(ncavg);
  Py_DECREF(avgdim);
  if (status != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "average: operand shapes incompatible with the bases/grid");
    return NULL;
  }
  Py_RETURN_NONE;
fail:
  Py_XDECREF(lo);
  Py_XDECREF(up);
  Py_XDECREF(nc);
  Py_XDECREF(ncavg);
  Py_XDECREF(avgdim);
  return NULL;
}

/* --------------------------------------------------------- pow(sqrt) */
static PyObject *
py_powsqrt(PyObject *self, PyObject *args)
{
  PyObject *bcap, *ncobj, *ocap, *acap;
  int num_quad;
  double exponent;
  if (!PyArg_ParseTuple(args, "OidOOO", &bcap, &num_quad, &exponent, &ncobj,
                        &ocap, &acap))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *out = array_arg(ocap), *a = array_arg(acap);
  if (!b || !out || !a)
    return NULL;
  PyArrayObject *nc =
      (PyArrayObject *)PyArray_FROM_OTF(ncobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (!nc)
    return NULL;
  int ndim = (int)PyArray_SIZE(nc);
  if (ndim < 1 || ndim > gpython_MAX_DIM) {
    Py_DECREF(nc);
    PyErr_SetString(PyExc_ValueError,
                    "powsqrt: cells must have 1 to 7 entries");
    return NULL;
  }
  int status =
      gpython_powsqrt(b, num_quad, exponent, ndim, PyArray_DATA(nc), out, a);
  Py_DECREF(nc);
  if (status != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "powsqrt: operand shapes incompatible with the basis/grid");
    return NULL;
  }
  Py_RETURN_NONE;
}

/* ------------------------------------------------------------- writing */
static PyObject *
py_write_field(PyObject *self, PyObject *args)
{
  const char *fname;
  PyObject *loobj, *upobj, *ncobj, *metaobj, *acap;
  if (!PyArg_ParseTuple(args, "sOOOOO", &fname, &loobj, &upobj, &ncobj,
                        &metaobj, &acap))
    return NULL;
  gpython_array *a = array_arg(acap);
  if (!a)
    return NULL;
  PyArrayObject *lo =
      (PyArrayObject *)PyArray_FROM_OTF(loobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *up =
      (PyArrayObject *)PyArray_FROM_OTF(upobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *nc =
      (PyArrayObject *)PyArray_FROM_OTF(ncobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (!lo || !up || !nc)
    goto fail;
  int ndim = (int)PyArray_SIZE(lo);
  if (ndim < 1 || ndim > gpython_MAX_DIM || PyArray_SIZE(up) != ndim ||
      PyArray_SIZE(nc) != ndim) {
    PyErr_SetString(PyExc_ValueError, "grid arrays must share ndim <= 7");
    goto fail;
  }
  const char *meta = NULL;
  Py_ssize_t meta_sz = 0;
  if (metaobj != Py_None) {
    if (PyBytes_AsStringAndSize(metaobj, (char **)&meta, &meta_sz) < 0)
      goto fail;
  }
  int status =
      gpython_write_field(fname, ndim, PyArray_DATA(lo), PyArray_DATA(up),
                          PyArray_DATA(nc), meta, (size_t)meta_sz, a);
  Py_DECREF(lo);
  Py_DECREF(up);
  Py_DECREF(nc);
  if (status == -1) {
    PyErr_SetString(PyExc_ValueError,
                    "write_field: grid cells do not cover the array");
    return NULL;
  }
  if (status != 0) {
    PyErr_Format(PyExc_OSError, "'%s': %s", fname, gpython_status_msg(status));
    return NULL;
  }
  Py_RETURN_NONE;
fail:
  Py_XDECREF(lo);
  Py_XDECREF(up);
  Py_XDECREF(nc);
  return NULL;
}

/* ------------------------------------------------------- differentiate */
static PyObject *
py_dg_differentiate(PyObject *self, PyObject *args)
{
  PyObject *bcap, *ocap, *icap;
  int dir, diff_order;
  double dx;
  if (!PyArg_ParseTuple(args, "OiidOO", &bcap, &dir, &diff_order, &dx, &ocap,
                        &icap))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *out = array_arg(ocap), *in = array_arg(icap);
  if (!b || !out || !in)
    return NULL;
  if (gpython_dg_differentiate(b, dir, diff_order, dx, out, in) != 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "dg_differentiate: operand shapes incompatible with the basis, or "
        "dir/diff_order out of range");
    return NULL;
  }
  Py_RETURN_NONE;
}

/* ---------------------------------------------------- evaluate-and-project */
static PyObject *
py_eval_at_coord_proj(PyObject *self, PyObject *args)
{
  PyObject *bcap, *loobj, *upobj, *ncobj, *evaldirsobj, *evalcoordsobj,
      *ncavgobj, *icap;
  int cdim_do, ndim_tar;
  if (!PyArg_ParseTuple(args, "OiOOOOOiOO", &bcap, &cdim_do, &loobj, &upobj,
                        &ncobj, &evaldirsobj, &evalcoordsobj, &ndim_tar,
                        &ncavgobj, &icap))
    return NULL;
  gpython_basis *b = basis_arg(bcap);
  gpython_array *in = array_arg(icap);
  if (!b || !in)
    return NULL;
  PyArrayObject *lo =
      (PyArrayObject *)PyArray_FROM_OTF(loobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *up =
      (PyArrayObject *)PyArray_FROM_OTF(upobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *nc =
      (PyArrayObject *)PyArray_FROM_OTF(ncobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *evaldirs = (PyArrayObject *)PyArray_FROM_OTF(
      evaldirsobj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *evalcoords = (PyArrayObject *)PyArray_FROM_OTF(
      evalcoordsobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *ncavg = (PyArrayObject *)PyArray_FROM_OTF(ncavgobj, NPY_INT32,
                                                           NPY_ARRAY_IN_ARRAY);
  if (!lo || !up || !nc || !evaldirs || !evalcoords || !ncavg)
    goto fail;
  int ndim = (int)PyArray_SIZE(lo);
  int num_eval = (int)PyArray_SIZE(evaldirs);
  if (ndim < 1 || ndim > gpython_MAX_DIM || PyArray_SIZE(up) != ndim ||
      PyArray_SIZE(nc) != ndim) {
    PyErr_SetString(PyExc_ValueError,
                    "eval_at_coord_proj: grid arrays must share ndim <= 7");
    goto fail;
  }
  if (PyArray_SIZE(evalcoords) != num_eval) {
    PyErr_SetString(
        PyExc_ValueError,
        "eval_at_coord_proj: eval_dirs and eval_coords must have the same "
        "length");
    goto fail;
  }
  if (ndim_tar < 1 || ndim_tar > gpython_MAX_DIM ||
      PyArray_SIZE(ncavg) != ndim_tar) {
    PyErr_SetString(
        PyExc_ValueError,
        "eval_at_coord_proj: cells_tar must have between 1 and 7 entries "
        "matching ndim_tar");
    goto fail;
  }
  int out_btype, out_poly_order, out_cdim, out_vdim;
  gpython_array *out = gpython_eval_at_coord_proj(
      b, cdim_do, ndim, PyArray_DATA(lo), PyArray_DATA(up), PyArray_DATA(nc),
      num_eval, PyArray_DATA(evaldirs), PyArray_DATA(evalcoords), ndim_tar,
      PyArray_DATA(ncavg), in, &out_btype, &out_poly_order, &out_cdim,
      &out_vdim);
  Py_DECREF(lo);
  Py_DECREF(up);
  Py_DECREF(nc);
  Py_DECREF(evaldirs);
  Py_DECREF(evalcoords);
  Py_DECREF(ncavg);
  if (!out) {
    PyErr_SetString(PyExc_ValueError,
                    "eval_at_coord_proj: operand shapes incompatible with the "
                    "basis/grid, or eval_dirs out of range");
    return NULL;
  }
  PyObject *outcap = wrap_array(out);
  if (!outcap)
    return NULL;
  return Py_BuildValue("(Niiii)", outcap, out_btype, out_poly_order, out_cdim,
                       out_vdim);
fail:
  Py_XDECREF(lo);
  Py_XDECREF(up);
  Py_XDECREF(nc);
  Py_XDECREF(evaldirs);
  Py_XDECREF(evalcoords);
  Py_XDECREF(ncavg);
  return NULL;
}

/* --------------------------------------------------------- dynvectors */
static PyObject *
py_dynvec_read(PyObject *self, PyObject *args)
{
  const char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname))
    return NULL;
  size_t ncomp;
  gpython_array *tm = NULL, *data = NULL;
  int status = gpython_dynvec_read(fname, &ncomp, &tm, &data);
  if (status != 0) {
    static const char *msgs[] = {
        "",
        "no such dynvector file (or empty/unrecognized header)",
        "dynvector is not double-precision (unsupported)",
        "failed to read dynvector data",
    };
    PyErr_Format(PyExc_OSError, "'%s': %s", fname,
                 msgs[status >= 1 && status <= 3 ? status : 0]);
    return NULL;
  }
  PyObject *tm_cap = wrap_array(tm);
  PyObject *data_cap = wrap_array(data);
  if (!tm_cap || !data_cap) {
    Py_XDECREF(tm_cap);
    Py_XDECREF(data_cap);
    return NULL;
  }
  return Py_BuildValue("(nNN)", (Py_ssize_t)ncomp, tm_cap, data_cap);
}

static PyObject *
py_dynvec_write(PyObject *self, PyObject *args)
{
  const char *fname;
  PyObject *tmobj, *dataobj;
  if (!PyArg_ParseTuple(args, "sOO", &fname, &tmobj, &dataobj))
    return NULL;
  PyArrayObject *tm =
      (PyArrayObject *)PyArray_FROM_OTF(tmobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *data = (PyArrayObject *)PyArray_FROM_OTF(dataobj, NPY_DOUBLE,
                                                          NPY_ARRAY_IN_ARRAY);
  if (!tm || !data) {
    Py_XDECREF(tm);
    Py_XDECREF(data);
    return NULL;
  }
  npy_intp n = PyArray_DIM(tm, 0);
  npy_intp ncomp = PyArray_NDIM(data) > 1 ? PyArray_DIM(data, 1) : 1;
  if (PyArray_DIM(data, 0) != n) {
    Py_DECREF(tm);
    Py_DECREF(data);
    PyErr_SetString(PyExc_ValueError,
                    "dynvec_write: tm and data must share the same length");
    return NULL;
  }
  /* gpython_dynvec_write (gkeyll/core/zero/dynvec.c) returns raw errno from its
   * fopen/fwrite calls, which is only set on failure and never cleared on
   * success -- so a stale errno from an earlier, unrelated failed syscall in
   * this process would otherwise be misread as this write having failed. */
  errno = 0;
  int status = gpython_dynvec_write(fname, (size_t)ncomp, (size_t)n,
                                    PyArray_DATA(tm), PyArray_DATA(data));
  Py_DECREF(tm);
  Py_DECREF(data);
  if (status != 0) {
    PyErr_Format(PyExc_OSError, "'%s': dynvector write failed", fname);
    return NULL;
  }
  Py_RETURN_NONE;
}

/* --------------------------------------------------------------- module */
static PyMethodDef gpython_methods[] = {
    {"api_version", py_api_version, METH_NOARGS, "gpython shim API version"},
    {"array_new", py_array_new, METH_VARARGS, "zeroed native array"},
    {"array_from_numpy", py_array_from_numpy, METH_VARARGS,
     "zero-copy native view of a (…, ncomp) float64 array"},
    {"array_clone", py_array_clone, METH_VARARGS, "deep copy"},
    {"array_ncomp", py_array_ncomp, METH_VARARGS, "components per cell"},
    {"array_size", py_array_size, METH_VARARGS, "number of cells"},
    {"array_view", py_array_view, METH_VARARGS,
     "read-only (size, ncomp) view pinning the native memory"},
    {"file_type", py_file_type, METH_VARARGS, "gkyl file type"},
    {"read_header", py_read_header, METH_VARARGS,
     "((ndim, lower, upper, cells), file_type, meta, esznc, tot_cells)"},
    {"read_field", py_read_field, METH_VARARGS,
     "((ndim, lower, upper, cells), array)"},
    {"basis_new", py_basis_new, METH_VARARGS, "basis handle"},
    {"basis_new_hybrid", py_basis_new_hybrid, METH_VARARGS,
     "basis handle (hybrid/gkhybrid, by cdim/vdim)"},
    {"basis_info", py_basis_info, METH_VARARGS,
     "(ndim, poly_order, num_basis, id)"},
    {"basis_eval", py_basis_eval, METH_VARARGS, "basis functions at a point"},
    {"basis_node_list", py_basis_node_list, METH_VARARGS,
     "(num_basis, ndim) node coordinates"},
    {"basis_nodal_to_modal", py_basis_nodal_to_modal, METH_VARARGS,
     "one-cell nodal -> modal"},
    {"dg_mul", py_dg_mul, METH_VARARGS, "weak product (per field)"},
    {"dg_div", py_dg_div, METH_VARARGS, "weak quotient (per field)"},
    {"dg_inv", py_dg_inv, METH_VARARGS, "weak reciprocal (per field)"},
    {"dg_mul_conf_phase", py_dg_mul_conf_phase, METH_VARARGS,
     "conf-space x phase-space weak product (single field)"},
    {"array_set", py_array_set, METH_VARARGS, "out = c*a"},
    {"array_accumulate", py_array_accumulate, METH_VARARGS, "out += c*a"},
    {"array_scale", py_array_scale, METH_VARARGS, "a *= c (in place)"},
    {"array_shiftc", py_array_shiftc, METH_VARARGS,
     "a[:, comp] += val (in place)"},
    {"array_reduce", py_array_reduce, METH_VARARGS,
     "per-component min/max/sum"},
    {"array_dg_reduce", py_array_dg_reduce, METH_VARARGS,
     "field-aware (Gauss-node) min/max/sum of one component"},
    {"array_integrate", py_array_integrate, METH_VARARGS,
     "int dx op(f) per field"},
    {"array_average", py_array_average, METH_VARARGS,
     "weighted (or plain) average over the flagged dims (single field)"},
    {"powsqrt", py_powsqrt, METH_VARARGS,
     "pow(sqrt(a), exponent) via gkyl_proj_powsqrt_on_basis (single field)"},
    {"dg_differentiate", py_dg_differentiate, METH_VARARGS,
     "local DG derivative (per field, no inter-cell stencil)"},
    {"eval_at_coord_proj", py_eval_at_coord_proj, METH_VARARGS,
     "evaluate at coords and project onto the lower-dim target basis; "
     "returns (array, target_btype, target_poly_order, target_cdim, "
     "target_vdim)"},
    {"write_field", py_write_field, METH_VARARGS,
     "write (lower, upper, cells, meta_bytes_or_None, array) to a .gkyl file"},
    {"dynvec_read", py_dynvec_read, METH_VARARGS,
     "(ncomp, tm_array, data_array) read from a dynvector file"},
    {"dynvec_write", py_dynvec_write, METH_VARARGS,
     "write a dynvector from parallel tm[n]/data[n,ncomp] arrays"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef gpython_module = {
    PyModuleDef_HEAD_INIT,
    "_gpython",
    "Compiled bridge to Gkeyll via the gpython shim (see GKEYLL_C_SHIM.md).",
    -1,
    gpython_methods,
};

PyMODINIT_FUNC
PyInit__gpython(void)
{
  import_array();
  PyObject *m = PyModule_Create(&gpython_module);
  if (!m)
    return NULL;
  if (PyModule_AddIntConstant(m, "GPYTHON_API_VERSION", GPYTHON_API_VERSION) <
      0) {
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
