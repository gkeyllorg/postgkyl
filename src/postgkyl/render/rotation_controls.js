const gd = document.getElementById('{plot_id}');
const sceneName = '__PGKYL_SCENE_NAME__';
const defaultAzimuthDeg = __PGKYL_AZIMUTH_DEG__;
const defaultPolarDeg = __PGKYL_POLAR_DEG__;
const defaultPeriodSec = __PGKYL_PERIOD_SEC__;
const defaultRadius = __PGKYL_RADIUS__;
let rafId = null;
let startMs = null;

let azimuthDeg = defaultAzimuthDeg;
let polarDeg = defaultPolarDeg;
let periodSec = defaultPeriodSec;
let cameraRadius = defaultRadius;

let theta0 = 0.0;
let omega = 0.0;
let xyRadius = 0.0;
let zEye = 0.0;

const clampPositive = (value, fallback) => (Number.isFinite(value) && value > 0.0 ? value : fallback);

const recomputeRotationParams = () => {
  const polarRad = polarDeg * Math.PI / 180.0;
  theta0 = azimuthDeg * Math.PI / 180.0;
  xyRadius = cameraRadius * Math.sin(polarRad);
  zEye = cameraRadius * Math.cos(polarRad);
  omega = 2.0 * Math.PI / periodSec;
};

const updateCamera = (theta) => {
  const camera = {
    eye: {x: xyRadius * Math.cos(theta), y: xyRadius * Math.sin(theta), z: zEye},
    up: {x: 0.0, y: 0.0, z: 1.0},
    center: {x: 0.0, y: 0.0, z: 0.0}
  };
  Plotly.relayout(gd, { [sceneName + '.camera']: camera });
};

const startRotation = () => {
  if (rafId === null) {
    rafId = requestAnimationFrame(animate);
  }
};

const stopRotation = () => {
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
};

const resetRotation = () => {
  startMs = null;
  updateCamera(theta0);
  startRotation();
};

const parent = gd.parentNode;
if (parent) {
  if (getComputedStyle(parent).position === 'static') {
    parent.style.position = 'relative';
  }

  const controls = document.createElement('div');
  controls.style.position = 'absolute';
  controls.style.top = '12px';
  controls.style.left = '12px';
  controls.style.zIndex = '20';
  controls.style.background = 'rgba(255, 255, 255, 0.92)';
  controls.style.border = '1px solid #b7bec8';
  controls.style.borderRadius = '8px';
  controls.style.padding = '8px 10px';
  controls.style.fontFamily = 'sans-serif';
  controls.style.fontSize = '12px';
  controls.style.color = '#1f2933';
  controls.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.18)';
  controls.style.display = 'grid';
  controls.style.gridTemplateColumns = 'auto auto';
  controls.style.gap = '6px 8px';
  controls.style.alignItems = 'center';
  controls.style.opacity = '0';
  controls.style.pointerEvents = 'none';
  controls.style.transition = 'opacity 120ms ease';

  const showControlsButton = document.createElement('button');
  showControlsButton.type = 'button';
  showControlsButton.textContent = 'Show rotation controls';
  showControlsButton.style.position = 'absolute';
  showControlsButton.style.top = '12px';
  showControlsButton.style.left = '12px';
  showControlsButton.style.zIndex = '21';
  showControlsButton.style.fontSize = '12px';
  showControlsButton.style.padding = '4px 8px';
  showControlsButton.style.cursor = 'pointer';
  showControlsButton.style.opacity = '0';
  showControlsButton.style.pointerEvents = 'none';
  showControlsButton.style.transition = 'opacity 120ms ease';

  const makeNumberInput = (value, min, step) => {
    const input = document.createElement('input');
    input.type = 'number';
    input.value = String(value);
    input.min = String(min);
    input.step = String(step);
    input.style.width = '86px';
    input.style.fontSize = '12px';
    return input;
  };

  const addRow = (labelText, inputEl) => {
    const label = document.createElement('label');
    label.textContent = labelText;
    controls.appendChild(label);
    controls.appendChild(inputEl);
  };

  const periodInput = makeNumberInput(defaultPeriodSec, 0.001, 0.1);
  const azimuthInput = makeNumberInput(defaultAzimuthDeg, -3600, 1);
  const polarInput = makeNumberInput(defaultPolarDeg, -3600, 1);
  const radiusInput = makeNumberInput(defaultRadius, 0.001, 0.1);

  addRow('Period (s)', periodInput);
  addRow('Azimuth (deg)', azimuthInput);
  addRow('Polar (deg)', polarInput);
  addRow('Radius', radiusInput);

  const buttonWrap = document.createElement('div');
  buttonWrap.style.gridColumn = '1 / span 2';
  buttonWrap.style.display = 'flex';
  buttonWrap.style.gap = '8px';

  const applyButton = document.createElement('button');
  applyButton.type = 'button';
  applyButton.textContent = 'Apply';

  const stopButton = document.createElement('button');
  stopButton.type = 'button';
  stopButton.textContent = 'Stop rotation';

  const hideButton = document.createElement('button');
  hideButton.type = 'button';
  hideButton.textContent = 'Hide controls';

  for (const btn of [applyButton, stopButton, hideButton]) {
    btn.style.fontSize = '12px';
    btn.style.padding = '3px 8px';
    btn.style.cursor = 'pointer';
  }

  let controlsCollapsed = true;
  let hoverActive = false;
  let hideTimer = null;

  const setControlsVisible = (visible) => {
    controls.style.opacity = visible ? '1' : '0';
    controls.style.pointerEvents = visible ? 'auto' : 'none';
  };

  const setShowButtonVisible = (visible) => {
    showControlsButton.style.opacity = visible ? '1' : '0';
    showControlsButton.style.pointerEvents = visible ? 'auto' : 'none';
  };

  const refreshControlsVisibility = () => {
    if (!hoverActive) {
      setControlsVisible(false);
      setShowButtonVisible(false);
      return;
    }
    if (controlsCollapsed) {
      setControlsVisible(false);
      setShowButtonVisible(true);
    } else {
      setControlsVisible(true);
      setShowButtonVisible(false);
    }
  };

  const clearHideTimer = () => {
    if (hideTimer !== null) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  };

  const scheduleHide = () => {
    clearHideTimer();
    hideTimer = setTimeout(() => {
      hoverActive = false;
      refreshControlsVisibility();
    }, 100);
  };

  const applyInputs = () => {
    periodSec = clampPositive(parseFloat(periodInput.value), defaultPeriodSec);
    cameraRadius = clampPositive(parseFloat(radiusInput.value), defaultRadius);
    azimuthDeg = Number.isFinite(parseFloat(azimuthInput.value)) ? parseFloat(azimuthInput.value) : defaultAzimuthDeg;
    polarDeg = Number.isFinite(parseFloat(polarInput.value)) ? parseFloat(polarInput.value) : defaultPolarDeg;

    periodInput.value = String(periodSec);
    radiusInput.value = String(cameraRadius);
    azimuthInput.value = String(azimuthDeg);
    polarInput.value = String(polarDeg);

    recomputeRotationParams();
    resetRotation();
  };

  applyButton.addEventListener('click', () => {
    applyInputs();
  });

  stopButton.addEventListener('click', () => {
    stopRotation();
  });

  hideButton.addEventListener('click', () => {
    controlsCollapsed = true;
    refreshControlsVisibility();
  });

  showControlsButton.addEventListener('click', () => {
    controlsCollapsed = false;
    hoverActive = true;
    refreshControlsVisibility();
  });

  parent.addEventListener('mouseenter', () => {
    hoverActive = true;
    clearHideTimer();
    refreshControlsVisibility();
  });

  parent.addEventListener('mouseleave', () => {
    scheduleHide();
  });

  buttonWrap.appendChild(applyButton);
  buttonWrap.appendChild(stopButton);
  buttonWrap.appendChild(hideButton);
  controls.appendChild(buttonWrap);
  parent.appendChild(controls);
  parent.appendChild(showControlsButton);
  refreshControlsVisibility();
}

gd.addEventListener('mousedown', stopRotation);
gd.addEventListener('wheel', stopRotation);
gd.addEventListener('touchstart', stopRotation);

const animate = (timestamp) => {
  if (startMs === null) {
    startMs = timestamp;
  }
  const elapsedSeconds = (timestamp - startMs) / 1000.0;
  const theta = theta0 + omega * elapsedSeconds;
  updateCamera(theta);
  rafId = requestAnimationFrame(animate);
};

recomputeRotationParams();
updateCamera(theta0);
startRotation();
