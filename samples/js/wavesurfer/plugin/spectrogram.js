/*!
 * wavesurfer.js 2.0.5 (Wed Apr 11 2018 18:25:13 GMT+0200 (CEST))
 * https://github.com/katspaugh/wavesurfer.js
 * @license BSD-3-Clause
 */
(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define("spectrogram", [], factory);
	else if(typeof exports === 'object')
		exports["spectrogram"] = factory();
	else
		root["WaveSurfer"] = root["WaveSurfer"] || {}, root["WaveSurfer"]["spectrogram"] = factory();
})(typeof self !== 'undefined' ? self : this, function() {
return /******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "localhost:8080/dist/plugin/";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 3);
/******/ })
/************************************************************************/
/******/ ({

/***/ 3:
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

/**
 * Calculate FFT - Based on https://github.com/corbanbrook/dsp.js
 */
/* eslint-disable complexity, no-redeclare, no-var, one-var */
var FFT = function FFT(bufferSize, sampleRate, windowFunc, alpha) {
    this.bufferSize = bufferSize;
    this.sampleRate = sampleRate;
    this.bandwidth = 2 / bufferSize * sampleRate / 2;

    this.sinTable = new Float32Array(bufferSize);
    this.cosTable = new Float32Array(bufferSize);
    this.windowValues = new Float32Array(bufferSize);
    this.reverseTable = new Uint32Array(bufferSize);

    this.peakBand = 0;
    this.peak = 0;

    switch (windowFunc) {
        case 'bartlett':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = 2 / (bufferSize - 1) * ((bufferSize - 1) / 2 - Math.abs(i - (bufferSize - 1) / 2));
            }
            break;
        case 'bartlettHann':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = 0.62 - 0.48 * Math.abs(i / (bufferSize - 1) - 0.5) - 0.38 * Math.cos(Math.PI * 2 * i / (bufferSize - 1));
            }
            break;
        case 'blackman':
            alpha = alpha || 0.16;
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = (1 - alpha) / 2 - 0.5 * Math.cos(Math.PI * 2 * i / (bufferSize - 1)) + alpha / 2 * Math.cos(4 * Math.PI * i / (bufferSize - 1));
            }
            break;
        case 'cosine':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = Math.cos(Math.PI * i / (bufferSize - 1) - Math.PI / 2);
            }
            break;
        case 'gauss':
            alpha = alpha || 0.25;
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = Math.pow(Math.E, -0.5 * Math.pow((i - (bufferSize - 1) / 2) / (alpha * (bufferSize - 1) / 2), 2));
            }
            break;
        case 'hamming':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = 0.54 - 0.46 * Math.cos(Math.PI * 2 * i / (bufferSize - 1));
            }
            break;
        case 'hann':
        case undefined:
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = 0.5 * (1 - Math.cos(Math.PI * 2 * i / (bufferSize - 1)));
            }
            break;
        case 'lanczoz':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = Math.sin(Math.PI * (2 * i / (bufferSize - 1) - 1)) / (Math.PI * (2 * i / (bufferSize - 1) - 1));
            }
            break;
        case 'rectangular':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = 1;
            }
            break;
        case 'triangular':
            for (var i = 0; i < bufferSize; i++) {
                this.windowValues[i] = 2 / bufferSize * (bufferSize / 2 - Math.abs(i - (bufferSize - 1) / 2));
            }
            break;
        default:
            throw Error("No such window function '" + windowFunc + "'");
    }

    var limit = 1;
    var bit = bufferSize >> 1;

    var i;

    while (limit < bufferSize) {
        for (i = 0; i < limit; i++) {
            this.reverseTable[i + limit] = this.reverseTable[i] + bit;
        }

        limit = limit << 1;
        bit = bit >> 1;
    }

    for (i = 0; i < bufferSize; i++) {
        this.sinTable[i] = Math.sin(-Math.PI / i);
        this.cosTable[i] = Math.cos(-Math.PI / i);
    }

    this.calculateSpectrum = function (buffer) {
        // Locally scope variables for speed up
        var bufferSize = this.bufferSize,
            cosTable = this.cosTable,
            sinTable = this.sinTable,
            reverseTable = this.reverseTable,
            real = new Float32Array(bufferSize),
            imag = new Float32Array(bufferSize),
            bSi = 2 / this.bufferSize,
            sqrt = Math.sqrt,
            rval,
            ival,
            mag,
            spectrum = new Float32Array(bufferSize / 2);

        var k = Math.floor(Math.log(bufferSize) / Math.LN2);

        if (Math.pow(2, k) !== bufferSize) {
            throw 'Invalid buffer size, must be a power of 2.';
        }
        if (bufferSize !== buffer.length) {
            throw 'Supplied buffer is not the same size as defined FFT. FFT Size: ' + bufferSize + ' Buffer Size: ' + buffer.length;
        }

        var halfSize = 1,
            phaseShiftStepReal,
            phaseShiftStepImag,
            currentPhaseShiftReal,
            currentPhaseShiftImag,
            off,
            tr,
            ti,
            tmpReal;

        for (var i = 0; i < bufferSize; i++) {
            real[i] = buffer[reverseTable[i]] * this.windowValues[reverseTable[i]];
            imag[i] = 0;
        }

        while (halfSize < bufferSize) {
            phaseShiftStepReal = cosTable[halfSize];
            phaseShiftStepImag = sinTable[halfSize];

            currentPhaseShiftReal = 1;
            currentPhaseShiftImag = 0;

            for (var fftStep = 0; fftStep < halfSize; fftStep++) {
                var i = fftStep;

                while (i < bufferSize) {
                    off = i + halfSize;
                    tr = currentPhaseShiftReal * real[off] - currentPhaseShiftImag * imag[off];
                    ti = currentPhaseShiftReal * imag[off] + currentPhaseShiftImag * real[off];

                    real[off] = real[i] - tr;
                    imag[off] = imag[i] - ti;
                    real[i] += tr;
                    imag[i] += ti;

                    i += halfSize << 1;
                }

                tmpReal = currentPhaseShiftReal;
                currentPhaseShiftReal = tmpReal * phaseShiftStepReal - currentPhaseShiftImag * phaseShiftStepImag;
                currentPhaseShiftImag = tmpReal * phaseShiftStepImag + currentPhaseShiftImag * phaseShiftStepReal;
            }

            halfSize = halfSize << 1;
        }

        for (var i = 0, N = bufferSize / 2; i < N; i++) {
            rval = real[i];
            ival = imag[i];
            mag = bSi * sqrt(rval * rval + ival * ival);

            if (mag > this.peak) {
                this.peakBand = i;
                this.peak = mag;
            }
            spectrum[i] = mag;
        }
        return spectrum;
    };
};
/* eslint-enable complexity, no-redeclare, no-var, one-var */

/**
 * @typedef {Object} SpectrogramPluginParams
 * @property {string|HTMLElement} container Selector of element or element in
 * which to render
 * @property {number} fftSamples=512 number of samples to fetch to FFT. Must be
 * a pwer of 2.
 * @property {number} noverlap Size of the overlapping window. Must be <
 * fftSamples. Auto deduced from canvas size by default.
 * @property {string} windowFunc='hann' The window function to be used. One of
 * these: `'bartlett'`, `'bartlettHann'`, `'blackman'`, `'cosine'`, `'gauss'`,
 * `'hamming'`, `'hann'`, `'lanczoz'`, `'rectangular'`, `'triangular'`
 * @property {?number} alpha Some window functions have this extra value.
 * (Between 0 and 1)
 * @property {number} pixelRatio=wavesurfer.params.pixelRatio to control the
 * size of the spectrogram in relation with its canvas. 1 = Draw on the whole
 * canvas. 2 = Draw on a quarter (1/2 the length and 1/2 the width)
 * @property {?boolean} deferInit Set to true to manually call
 * `initPlugin('spectrogram')`
 */

/**
 * Render a spectrogram visualisation of the audio.
 *
 * @implements {PluginClass}
 * @extends {Observer}
 * @example
 * // es6
 * import SpectrogramPlugin from 'wavesurfer.spectrogram.js';
 *
 * // commonjs
 * var SpectrogramPlugin = require('wavesurfer.spectrogram.js');
 *
 * // if you are using <script> tags
 * var SpectrogramPlugin = window.WaveSurfer.spectrogram;
 *
 * // ... initialising wavesurfer with the plugin
 * var wavesurfer = WaveSurfer.create({
 *   // wavesurfer options ...
 *   plugins: [
 *     SpectrogramPlugin.create({
 *       // plugin options ...
 *     })
 *   ]
 * });
 */

var SpectrogramPlugin = function () {
    _createClass(SpectrogramPlugin, null, [{
        key: 'create',

        /**
         * Spectrogram plugin definition factory
         *
         * This function must be used to create a plugin definition which can be
         * used by wavesurfer to correctly instantiate the plugin.
         *
         * @param  {SpectrogramPluginParams} params parameters use to initialise the plugin
         * @return {PluginDefinition} an object representing the plugin
         */
        value: function create(params) {
            return {
                name: 'spectrogram',
                deferInit: params && params.deferInit ? params.deferInit : false,
                params: params,
                staticProps: {
                    FFT: FFT
                },
                instance: SpectrogramPlugin
            };
        }
    }]);

    function SpectrogramPlugin(params, ws) {
        var _this = this;

        _classCallCheck(this, SpectrogramPlugin);

        this.params = params;
        this.wavesurfer = ws;
        this.util = ws.util;

        this.frequenciesDataUrl = params.frequenciesDataUrl;
        this._onScroll = function (e) {
            _this.updateScroll(e);
        };
        this._onReady = function () {
            var drawer = _this.drawer = ws.drawer;

            _this.container = 'string' == typeof params.container ? document.querySelector(params.container) : params.container;

            if (!_this.container) {
                throw Error('No container for WaveSurfer spectrogram');
            }

            _this.width = drawer.width;
            _this.pixelRatio = _this.params.pixelRatio || ws.params.pixelRatio;
            _this.fftSamples = _this.params.fftSamples || ws.params.fftSamples || 512;
            _this.height = _this.fftSamples / 2;
            _this.noverlap = params.noverlap;
            _this.windowFunc = params.windowFunc;
            _this.alpha = params.alpha;

            _this.createWrapper();
            _this.createCanvas();
            _this.render();

            drawer.wrapper.addEventListener('scroll', _this._onScroll);
            ws.on('redraw', function () {
                return _this.render();
            });
        };
    }

    _createClass(SpectrogramPlugin, [{
        key: 'init',
        value: function init() {
            // Check if ws is ready
            if (this.wavesurfer.isReady) {
                this._onReady();
            }

            this.wavesurfer.on('ready', this._onReady);
        }
    }, {
        key: 'destroy',
        value: function destroy() {
            this.unAll();
            this.wavesurfer.un('ready', this._onReady);
            this.drawer.wrapper.removeEventListener('scroll', this._onScroll);
            this.wavesurfer = null;
            this.util = null;
            this.params = null;
            if (this.wrapper) {
                this.wrapper.parentNode.removeChild(this.wrapper);
                this.wrapper = null;
            }
        }
    }, {
        key: 'createWrapper',
        value: function createWrapper() {
            var _this2 = this;

            var prevSpectrogram = this.container.querySelector('spectrogram');
            if (prevSpectrogram) {
                this.container.removeChild(prevSpectrogram);
            }
            var wsParams = this.wavesurfer.params;
            this.wrapper = document.createElement('spectrogram');
            // if labels are active
            if (this.params.labels) {
                var labelsEl = this.labelsEl = document.createElement('canvas');
                labelsEl.classList.add('spec-labels');
                this.drawer.style(labelsEl, {
                    left: 0,
                    position: 'absolute',
                    zIndex: 9,
                    height: this.height / this.pixelRatio + 'px',
                    width: 55 / this.pixelRatio + 'px'
                });
                this.wrapper.appendChild(labelsEl);
                // can be customized in next version
                this.loadLabels('rgba(68,68,68,0.5)', '12px', '10px', '', '#fff', '#f7f7f7', 'center', '#specLabels');
            }

            this.drawer.style(this.wrapper, {
                display: 'block',
                position: 'relative',
                userSelect: 'none',
                webkitUserSelect: 'none',
                height: this.height / this.pixelRatio + 'px'
            });

            if (wsParams.fillParent || wsParams.scrollParent) {
                this.drawer.style(this.wrapper, {
                    width: '100%',
                    overflowX: 'hidden',
                    overflowY: 'hidden'
                });
            }
            this.container.appendChild(this.wrapper);

            this.wrapper.addEventListener('click', function (e) {
                e.preventDefault();
                var relX = 'offsetX' in e ? e.offsetX : e.layerX;
                _this2.fireEvent('click', relX / _this2.scrollWidth || 0);
            });
        }
    }, {
        key: 'createCanvas',
        value: function createCanvas() {
            var canvas = this.canvas = this.wrapper.appendChild(document.createElement('canvas'));

            this.spectrCc = canvas.getContext('2d');

            this.util.style(canvas, {
                position: 'absolute',
                zIndex: 4
            });
        }
    }, {
        key: 'render',
        value: function render() {
            this.updateCanvasStyle();

            if (this.frequenciesDataUrl) {
                this.loadFrequenciesData(this.frequenciesDataUrl);
            } else {
                this.getFrequencies(this.drawSpectrogram);
            }
        }
    }, {
        key: 'updateCanvasStyle',
        value: function updateCanvasStyle() {
            var width = Math.round(this.width / this.pixelRatio) + 'px';
            this.canvas.width = this.width;
            this.canvas.height = this.height;
            this.canvas.style.width = width;
        }
    }, {
        key: 'drawSpectrogram',
        value: function drawSpectrogram(frequenciesData, my) {
            var spectrCc = my.spectrCc;
            var length = my.wavesurfer.backend.getDuration();
            var height = my.height;
            var pixels = my.resample(frequenciesData);
            var heightFactor = my.buffer ? 2 / my.buffer.numberOfChannels : 1;
            var i = void 0;
            var j = void 0;

            for (i = 0; i < pixels.length; i++) {
                for (j = 0; j < pixels[i].length; j++) {
                    var colorValue = 255 - pixels[i][j];
                    my.spectrCc.fillStyle = 'rgb(' + colorValue + ', ' + colorValue + ', ' + colorValue + ')';
                    my.spectrCc.fillRect(i, height - j * heightFactor, 1, heightFactor);
                }
            }
        }
    }, {
        key: 'getFrequencies',
        value: function getFrequencies(callback) {
            var fftSamples = this.fftSamples;
            var buffer = this.buffer = this.wavesurfer.backend.buffer;
            var channelOne = buffer.getChannelData(0);
            var bufferLength = buffer.length;
            var sampleRate = buffer.sampleRate;
            var frequencies = [];

            if (!buffer) {
                this.fireEvent('error', 'Web Audio buffer is not available');
                return;
            }

            var noverlap = this.noverlap;
            if (!noverlap) {
                var uniqueSamplesPerPx = buffer.length / this.canvas.width;
                noverlap = Math.max(0, Math.round(fftSamples - uniqueSamplesPerPx));
            }

            var fft = new FFT(fftSamples, sampleRate, this.windowFunc, this.alpha);
            var maxSlicesCount = Math.floor(bufferLength / (fftSamples - noverlap));
            var currentOffset = 0;

            while (currentOffset + fftSamples < channelOne.length) {
                var segment = channelOne.slice(currentOffset, currentOffset + fftSamples);
                var spectrum = fft.calculateSpectrum(segment);
                var array = new Uint8Array(fftSamples / 2);
                var j = void 0;
                for (j = 0; j < fftSamples / 2; j++) {
                    array[j] = Math.max(-255, Math.log10(spectrum[j]) * 45);
                }
                frequencies.push(array);
                currentOffset += fftSamples - noverlap;
            }
            callback(frequencies, this);
        }
    }, {
        key: 'loadFrequenciesData',
        value: function loadFrequenciesData(url) {
            var _this3 = this;

            var ajax = this.util.ajax({ url: url });

            ajax.on('success', function (data) {
                return _this3.drawSpectrogram(JSON.parse(data), _this3);
            });
            ajax.on('error', function (e) {
                return _this3.fireEvent('error', 'XHR error: ' + e.target.statusText);
            });

            return ajax;
        }
    }, {
        key: 'freqType',
        value: function freqType(freq) {
            return freq >= 1000 ? (freq / 1000).toFixed(1) : Math.round(freq);
        }
    }, {
        key: 'unitType',
        value: function unitType(freq) {
            return freq >= 1000 ? 'KHz' : 'Hz';
        }
    }, {
        key: 'loadLabels',
        value: function loadLabels(bgFill, fontSizeFreq, fontSizeUnit, fontType, textColorFreq, textColorUnit, textAlign, container) {
            var frequenciesHeight = this.height;
            bgFill = bgFill || 'rgba(68,68,68,0)';
            fontSizeFreq = fontSizeFreq || '12px';
            fontSizeUnit = fontSizeUnit || '10px';
            fontType = fontType || 'Helvetica';
            textColorFreq = textColorFreq || '#fff';
            textColorUnit = textColorUnit || '#fff';
            textAlign = textAlign || 'center';
            container = container || '#specLabels';
            var getMaxY = frequenciesHeight || 512;
            var labelIndex = 5 * (getMaxY / 256);
            var freqStart = 0;
            var step = (this.wavesurfer.backend.ac.sampleRate / 2 - freqStart) / labelIndex;

            var ctx = this.labelsEl.getContext('2d');
            this.labelsEl.height = this.height;
            this.labelsEl.width = 55;

            ctx.fillStyle = bgFill;
            ctx.fillRect(0, 0, 55, getMaxY);
            ctx.fill();
            var i = void 0;

            for (i = 0; i <= labelIndex; i++) {
                ctx.textAlign = textAlign;
                ctx.textBaseline = 'middle';

                var freq = freqStart + step * i;
                var index = Math.round(freq / (this.sampleRate / 2) * this.fftSamples);
                var label = this.freqType(freq);
                var units = this.unitType(freq);
                var x = 16;
                var yLabelOffset = 2;

                if (i == 0) {
                    ctx.fillStyle = textColorUnit;
                    ctx.font = fontSizeUnit + ' ' + fontType;
                    ctx.fillText(units, x + 24, getMaxY + i - 10);
                    ctx.fillStyle = textColorFreq;
                    ctx.font = fontSizeFreq + ' ' + fontType;
                    ctx.fillText(label, x, getMaxY + i - 10);
                } else {
                    ctx.fillStyle = textColorUnit;
                    ctx.font = fontSizeUnit + ' ' + fontType;
                    ctx.fillText(units, x + 24, getMaxY - i * 50 + yLabelOffset);
                    ctx.fillStyle = textColorFreq;
                    ctx.font = fontSizeFreq + ' ' + fontType;
                    ctx.fillText(label, x, getMaxY - i * 50 + yLabelOffset);
                }
            }
        }
    }, {
        key: 'updateScroll',
        value: function updateScroll(e) {
            if (this.wrapper) {
                this.wrapper.scrollLeft = e.target.scrollLeft;
            }
        }
    }, {
        key: 'resample',
        value: function resample(oldMatrix) {
            var columnsNumber = this.width;
            var newMatrix = [];

            var oldPiece = 1 / oldMatrix.length;
            var newPiece = 1 / columnsNumber;
            var i = void 0;

            for (i = 0; i < columnsNumber; i++) {
                var column = new Array(oldMatrix[0].length);
                var j = void 0;

                for (j = 0; j < oldMatrix.length; j++) {
                    var oldStart = j * oldPiece;
                    var oldEnd = oldStart + oldPiece;
                    var newStart = i * newPiece;
                    var newEnd = newStart + newPiece;

                    var overlap = oldEnd <= newStart || newEnd <= oldStart ? 0 : Math.min(Math.max(oldEnd, newStart), Math.max(newEnd, oldStart)) - Math.max(Math.min(oldEnd, newStart), Math.min(newEnd, oldStart));
                    var k = void 0;
                    /* eslint-disable max-depth */
                    if (overlap > 0) {
                        for (k = 0; k < oldMatrix[0].length; k++) {
                            if (column[k] == null) {
                                column[k] = 0;
                            }
                            column[k] += overlap / newPiece * oldMatrix[j][k];
                        }
                    }
                    /* eslint-enable max-depth */
                }

                var intColumn = new Uint8Array(oldMatrix[0].length);
                var m = void 0;

                for (m = 0; m < oldMatrix[0].length; m++) {
                    intColumn[m] = column[m];
                }

                newMatrix.push(intColumn);
            }

            return newMatrix;
        }
    }]);

    return SpectrogramPlugin;
}();

exports.default = SpectrogramPlugin;
module.exports = exports['default'];

/***/ })

/******/ });
});
//# sourceMappingURL=wavesurfer.spectrogram.js.map