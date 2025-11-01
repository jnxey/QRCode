/*
qr-lib.js
A small, dependency-light browser library that wraps two battle-tested OSS projects:
 - qrcode-generator (kazuhikoarase) for encoding. (MIT)
 - jsQR (cozmo) for decoding. (Apache-2.0)

This file expects those libraries to be loaded (CDN examples below). The library exposes a small API:

  QRLib.generateToCanvas(text, {typeNumber, errorCorrectLevel, scale, margin}) -> HTMLCanvasElement
  QRLib.generateDataURL(text, opts) -> Promise<string> (dataURL)
  QRLib.decodeFromImageElement(imgEl) -> Promise<{data, location}|null>
  QRLib.decodeFromFile(file) -> Promise<{data, location}|null>
  QRLib.scanFromVideo(videoEl, onResult) -> {stop: fn}

CDN / usage (put in <head> or before this script):
<script src="https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js"></script>
<script src="https://unpkg.com/jsqr/dist/jsQR.js"></script>

License: use as you like. This is glue code only.
*/

const QRLib = (function () {
  // defaults
  const DEFAULTS = {
    typeNumber: 0, // 0 = automatic
    errorCorrectLevel: 'L', // 'L','M','Q','H'
    scale: 4, // pixels per QR module
    margin: 4 // quiet zone modules
  };

  function _merge(opts) {
    return Object.assign({}, DEFAULTS, opts || {});
  }

  function _getEC(level) {
    // map char to qrcode-generator constant if available
    if (typeof window.QRCode === 'undefined' && typeof window.qrcode === 'undefined') {
      // library exposes `qrcode` or `QRCode` depending on build. We don't crash here.
    }
    const map = { L: 'L', M: 'M', Q: 'Q', H: 'H' };
    return map[level] || 'L';
  }

  function generateToCanvas(text, opts) {
    const o = _merge(opts);
    // build qrcode using qrcode-generator API (kazuhikoarase)
    // API: qrcode(typeNumber, errorCorrectLevel)
    const QR = (typeof qrcode !== 'undefined') ? qrcode : (typeof QRCode !== 'undefined' ? QRCode : null);
    if (!QR) throw new Error('qrcode-generator library not found. Include it via CDN.');

    const qr = QR(o.typeNumber, o.errorCorrectLevel || _getEC(o.errorCorrectLevel));
    qr.addData(String(text));
    qr.make();

    const moduleCount = qr.getModuleCount();
    const canvas = document.createElement('canvas');
    const size = (moduleCount + 2 * o.margin) * o.scale;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, size, size);

    ctx.fillStyle = '#000000';
    for (let row = 0; row < moduleCount; row++) {
      for (let col = 0; col < moduleCount; col++) {
        if (qr.isDark(row, col)) {
          const w = Math.ceil((col + o.margin) * o.scale);
          const h = Math.ceil((row + o.margin) * o.scale);
          ctx.fillRect(w, h, o.scale, o.scale);
        }
      }
    }

    return canvas;
  }

  function generateDataURL(text, opts) {
    return Promise.resolve().then(() => {
      const c = generateToCanvas(text, opts);
      return c.toDataURL('image/png');
    });
  }

  function _getImageDataFromImageElement(imgEl) {
    const canvas = document.createElement('canvas');
    canvas.width = imgEl.naturalWidth || imgEl.width;
    canvas.height = imgEl.naturalHeight || imgEl.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  function decodeFromImageElement(imgEl) {
    return new Promise((resolve, reject) => {
      if (typeof jsQR === 'undefined') {
        reject(new Error('jsQR library not found. Include it via CDN.'));
        return;
      }
      try {
        const imgData = _getImageDataFromImageElement(imgEl);
        const code = jsQR(imgData.data, imgData.width, imgData.height);
        resolve(code || null);
      } catch (err) {
        reject(err);
      }
    });
  }

  function decodeFromFile(file) {
    return new Promise((resolve, reject) => {
      if (!(file instanceof File)) {
        reject(new Error('Expected a File object'));
        return;
      }
      const img = new Image();
      img.onload = () => {
        decodeFromImageElement(img).then(resolve, reject);
      };
      img.onerror = (e) => reject(new Error('Failed to load image file'));
      const reader = new FileReader();
      reader.onload = (ev) => {
        img.src = ev.target.result;
      };
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(file);
    });
  }

  function scanFromVideo(videoEl, onResult, scanInterval = 200) {
    if (typeof jsQR === 'undefined') throw new Error('jsQR library not found.');
    let running = true;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    async function tick() {
      if (!running) return;
      if (videoEl.readyState >= 2) {
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const code = jsQR(imgData.data, imgData.width, imgData.height);
        if (code) onResult(code);
      }
      setTimeout(tick, scanInterval);
    }

    tick();
    return {
      stop() { running = false; }
    };
  }

  return {
    generateToCanvas,
    generateDataURL,
    decodeFromImageElement,
    decodeFromFile,
    scanFromVideo
  };
})();

// export for module systems
if (typeof module !== 'undefined') module.exports = QRLib;
