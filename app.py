import streamlit as st
import streamlit.components.v1 as components

# 페이지 기본 설정
st.set_page_config(
    page_title="머신러닝 플레이그라운드 - 선형 회귀",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 스트림릿 페이지 스타일 최적화 (여백 제거)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# HTML/CSS/JS 프론트엔드 코드
html_code = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>머신러닝 플레이그라운드 - 선형 회귀</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Google Font: Noto Sans KR -->
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Noto Sans KR', sans-serif;
      touch-action: manipulation;
    }
    
    .canvas-container {
      position: relative;
      width: 100%;
      padding-bottom: 60%; /* 비율을 조정하여 화면 가독성 개선 */
      height: 0;
    }
    
    .canvas-container canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }

    input[type=range]::-webkit-slider-thumb {
      width: 22px;
      height: 22px;
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen pb-12">

  <div class="max-w-4xl mx-auto px-4 py-4">
    
    <!-- 헤더 영역 -->
    <header class="text-center mb-6">
      <div class="inline-flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-bold mb-2">
        <span>고등학생을 위한 인공지능 실습</span>
        <span>•</span>
        <span>1단계</span>
      </div>
      <h1 class="text-2xl md:text-3xl font-extrabold text-slate-900 tracking-tight">
        🤖 머신러닝 플레이그라운드
      </h1>
      <p class="text-slate-600 text-sm md:text-base mt-1">
        점을 찍고 경사하강법으로 최적의 회귀선(y = mx + b)을 찾아내는 과정을 체험해보세요.
      </p>
    </header>

    <!-- 메인 카드 컨테이너 -->
    <div class="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden mb-6">
      
      <!-- 상단 컨트롤 바 -->
      <div class="p-4 md:p-5 bg-slate-100/80 border-b border-slate-200 flex flex-col md:flex-row gap-4 justify-between items-stretch md:items-center">
        
        <!-- 모드 전환 버튼 -->
        <div class="flex bg-slate-200/80 p-1 rounded-xl gap-1">
          <button id="modeAddBtn" onclick="setMode('add')" class="flex-1 md:flex-none px-4 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 shadow-sm bg-white text-blue-600">
            <span>📍 데이터 추가</span>
          </button>
          <button id="modePredictBtn" onclick="setMode('predict')" class="flex-1 md:flex-none px-4 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 text-slate-600 hover:text-slate-900">
            <span>🎯 예측하기</span>
          </button>
        </div>

        <!-- 액션 버튼 그룹 -->
        <div class="flex flex-wrap items-center gap-2">
          <button onclick="loadSampleData()" class="flex-1 sm:flex-none px-3.5 py-2.5 bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold rounded-xl text-sm transition-all active:scale-95 flex items-center justify-center gap-1">
            🎲 예시 데이터
          </button>
          <button id="trainBtn" onclick="startTraining()" class="flex-1 sm:flex-none px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl text-sm shadow-md shadow-blue-500/20 transition-all active:scale-95 flex items-center justify-center gap-1">
            ▶ 학습 시작
          </button>
          <button onclick="resetData()" class="px-3.5 py-2.5 bg-rose-100 hover:bg-rose-200 text-rose-700 font-semibold rounded-xl text-sm transition-all active:scale-95 flex items-center justify-center gap-1">
            🔄 초기화
          </button>
        </div>
      </div>

      <!-- 슬라이더 조절바 -->
      <div class="px-4 py-3 bg-slate-50 border-b border-slate-200 flex flex-col sm:flex-row gap-4 items-stretch sm:items-center justify-between text-sm">
        <div class="flex items-center gap-3 flex-1">
          <label for="epochsSlider" class="font-bold text-slate-700 whitespace-nowrap min-w-[90px]">
            학습 횟수 (Epochs):
          </label>
          <input type="range" id="epochsSlider" min="10" max="500" step="10" value="100" class="w-full accent-blue-600 cursor-pointer h-2 bg-slate-200 rounded-lg">
          <span id="epochsVal" class="font-extrabold text-blue-600 min-w-[45px] text-right">100회</span>
        </div>

        <div class="flex items-center gap-3 flex-1">
          <label for="lrSlider" class="font-bold text-slate-700 whitespace-nowrap min-w-[90px]">
            학습률 (Alpha):
          </label>
          <input type="range" id="lrSlider" min="0.005" max="0.2" step="0.005" value="0.05" class="w-full accent-blue-600 cursor-pointer h-2 bg-slate-200 rounded-lg">
          <span id="lrVal" class="font-extrabold text-blue-600 min-w-[50px] text-right">0.050</span>
        </div>
      </div>

      <!-- 가이드 메시지 바 -->
      <div id="guideBanner" class="px-4 py-2.5 bg-amber-50 border-b border-amber-200 text-amber-800 text-xs sm:text-sm font-medium flex items-center gap-2">
        <span>💡</span>
        <span id="guideText">캔버스를 터치하거나 클릭하여 데이터 점을 추가해보세요!</span>
      </div>

      <!-- 도출된 회귀선 방정식 표시 바 -->
      <div id="equationBox" class="px-4 py-2.5 bg-blue-50/90 border-b border-blue-200 flex items-center justify-between text-xs sm:text-sm font-semibold text-blue-900">
        <div class="flex items-center gap-2 flex-wrap">
          <span class="text-blue-600 font-bold flex items-center gap-1">
            <span>📐</span>
            <span>회귀선 방정식:</span>
          </span>
          <code id="equationText" class="px-2.5 py-1 bg-white border border-blue-300 rounded-lg font-mono text-blue-700 font-bold text-sm sm:text-base shadow-sm">
            ŷ = 0.000x + 50.000
          </code>
        </div>
        <span class="text-slate-600 text-xs hidden sm:inline-block bg-white/60 px-2 py-0.5 rounded border border-slate-200">
          학습 진행에 따라 실시간 갱신
        </span>
      </div>

      <!-- 캔버스 영역 -->
      <div class="p-4 bg-slate-900/5">
        <div class="canvas-container bg-white rounded-xl border border-slate-300 shadow-inner overflow-hidden cursor-crosshair">
          <canvas id="mlCanvas"></canvas>
        </div>
      </div>

    </div>

    <!-- 지표 표시 카드 그리드 -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
      
      <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">평균제곱오차 (MSE)</div>
        <div id="metricMSE" class="text-xl md:text-2xl font-black text-rose-600">-</div>
        <div class="text-[11px] text-slate-600 mt-1">오차의 제곱 평균 (작을수록 좋음)</div>
      </div>

      <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">기울기 (m)</div>
        <div id="metricSlope" class="text-xl md:text-2xl font-black text-blue-600">0.000</div>
        <div class="text-[11px] text-slate-600 mt-1">변화 비율 (y = mx + b)</div>
      </div>

      <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">절편 (b)</div>
        <div id="metricIntercept" class="text-xl md:text-2xl font-black text-indigo-600">0.000</div>
        <div class="text-[11px] text-slate-600 mt-1">x=0 일 때의 y 값</div>
      </div>

      <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm text-center">
        <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">결정계수 (R²)</div>
        <div id="metricR2" class="text-xl md:text-2xl font-black text-emerald-600">-</div>
        <div class="text-[11px] text-slate-600 mt-1">설명력 (1.0에 가까울수록 좋음)</div>
      </div>

    </div>

    <!-- 머신러닝 학습 가이드 카드 -->
    <div class="bg-white rounded-xl border border-slate-200 p-5 shadow-sm text-slate-700 text-sm">
      <h3 class="font-bold text-slate-900 mb-2 flex items-center gap-1.5 text-base">
        <span>📖</span> 선형 회귀(Linear Regression) 핵심 개념
      </h3>
      <ul class="space-y-1.5 list-disc list-inside text-xs sm:text-sm text-slate-600">
        <li><b>회귀선 방정식:</b> ŷ = mx + b (입력 x에 대한 예측값 ŷ를 직선으로 모델링)</li>
        <li><b>경사하강법(Gradient Descent):</b> 오차(MSE)를 줄이기 위해 기울기(m)와 절편(b)을 조금씩 수정하는 학습 알고리즘</li>
        <li><b>예측 모드:</b> 학습이 완료된 회귀선 위에 임의의 x값을 지정하여 모델이 y값을 어떻게 추정하는지 확인</li>
      </ul>
    </div>

  </div>

  <script>
    const canvas = document.getElementById('mlCanvas');
    const ctx = canvas.getContext('2d');

    let dataPoints = [];
    let m = 0;
    let b = 50;
    
    let currentMode = 'add';
    let isTraining = false;
    let predictedX = null;
    let animationFrameId = null;

    const epochsSlider = document.getElementById('epochsSlider');
    const epochsVal = document.getElementById('epochsVal');
    const lrSlider = document.getElementById('lrSlider');
    const lrVal = document.getElementById('lrVal');
    const guideBanner = document.getElementById('guideBanner');
    const guideText = document.getElementById('guideText');
    const trainBtn = document.getElementById('trainBtn');

    function resizeCanvas() {
      const container = canvas.parentElement;
      const rect = container.getBoundingClientRect();
      
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      
      canvas.cssWidth = rect.width;
      canvas.cssHeight = rect.height;

      render();
    }

    function dataToCanvasCoords(dataX, dataY) {
      const margin = 35;
      const width = canvas.cssWidth - margin * 2;
      const height = canvas.cssHeight - margin * 2;

      const px = margin + (dataX / 100) * width;
      const py = (canvas.cssHeight - margin) - (dataY / 100) * height;
      return { x: px, y: py };
    }

    function canvasToDataCoords(px, py) {
      const margin = 35;
      const width = canvas.cssWidth - margin * 2;
      const height = canvas.cssHeight - margin * 2;

      const dataX = ((px - margin) / width) * 100;
      const dataY = (((canvas.cssHeight - margin) - py) / height) * 100;

      return {
        x: Math.max(0, Math.min(100, dataX)),
        y: Math.max(0, Math.min(100, dataY))
      };
    }

    function calculateMetrics() {
      if (dataPoints.length === 0) {
        return { mse: null, r2: null };
      }

      let sumSquaredErrors = 0;
      let sumY = 0;

      for (const p of dataPoints) {
        const predY = m * p.x + b;
        const err = p.y - predY;
        sumSquaredErrors += err * err;
        sumY += p.y;
      }

      const mse = sumSquaredErrors / dataPoints.length;
      const meanY = sumY / dataPoints.length;

      let totalSumOfSquares = 0;
      for (const p of dataPoints) {
        totalSumOfSquares += Math.pow(p.y - meanY, 2);
      }

      let r2 = totalSumOfSquares === 0 ? 1 : 1 - (sumSquaredErrors / totalSumOfSquares);
      r2 = Math.max(-1, Math.min(1, r2));

      return { mse, r2 };
    }

    function performGradientDescentStep(learningRate) {
      if (dataPoints.length === 0) return;

      let gradM = 0;
      let gradB = 0;
      const N = dataPoints.length;

      for (const p of dataPoints) {
        const xNorm = p.x / 100;
        const yNorm = p.y / 100;
        const bNorm = b / 100;

        const predYNorm = m * xNorm + bNorm;
        const diff = predYNorm - yNorm;

        gradM += (2 / N) * xNorm * diff;
        gradB += (2 / N) * diff;
      }

      m -= learningRate * gradM;
      let bNorm = (b / 100) - learningRate * gradB;
      b = bNorm * 100;

      if (!isFinite(m)) m = 0;
      if (!isFinite(b)) b = 50;
    }

    function startTraining() {
      if (dataPoints.length < 2) {
        alert('최소 2개 이상의 데이터 점이 필요합니다!');
        return;
      }

      if (isTraining) return;
      isTraining = true;

      trainBtn.disabled = true;
      trainBtn.classList.add('opacity-50', 'cursor-not-allowed');

      const targetEpochs = parseInt(epochsSlider.value);
      const learningRate = parseFloat(lrSlider.value);
      let currentStep = 0;

      function step() {
        const stepsPerFrame = Math.max(1, Math.floor(targetEpochs / 50));
        
        for (let i = 0; i < stepsPerFrame && currentStep < targetEpochs; i++) {
          performGradientDescentStep(learningRate);
          currentStep++;
        }

        render();

        if (currentStep < targetEpochs) {
          animationFrameId = requestAnimationFrame(step);
        } else {
          isTraining = false;
          trainBtn.disabled = false;
          trainBtn.classList.remove('opacity-50', 'cursor-not-allowed');
          
          guideBanner.className = "px-4 py-2.5 bg-emerald-50 border-b border-emerald-200 text-emerald-800 text-xs sm:text-sm font-medium flex items-center gap-2";
          guideText.innerText = "🎉 학습이 완료되었습니다! 결정계수(R²)와 오차(MSE)를 확인해보세요.";
        }
      }

      step();
    }

    function render() {
      if (!canvas.cssWidth) return;

      const w = canvas.cssWidth;
      const h = canvas.cssHeight;
      const margin = 35;

      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = '#f1f5f9';
      ctx.lineWidth = 1;

      for (let i = 0; i <= 100; i += 10) {
        const pt = dataToCanvasCoords(i, i);
        ctx.beginPath();
        ctx.moveTo(pt.x, margin);
        ctx.lineTo(pt.x, h - margin);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(margin, pt.y);
        ctx.lineTo(w - margin, pt.y);
        ctx.stroke();
      }

      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(margin, h - margin);
      ctx.lineTo(w - margin, h - margin);
      ctx.moveTo(margin, margin);
      ctx.lineTo(margin, h - margin);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      
      [0, 50, 100].forEach(val => {
        const pt = dataToCanvasCoords(val, val);
        ctx.fillText(val.toString(), pt.x, h - margin + 16);
        ctx.textAlign = 'right';
        ctx.fillText(val.toString(), margin - 8, pt.y + 4);
        ctx.textAlign = 'center';
      });

      const startPt = dataToCanvasCoords(0, b);
      const endPt = dataToCanvasCoords(100, m * 100 + b);

      ctx.save();
      ctx.beginPath();
      ctx.rect(margin, margin, w - margin * 2, h - margin * 2);
      ctx.clip();

      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(startPt.x, startPt.y);
      ctx.lineTo(endPt.x, endPt.y);
      ctx.stroke();

      ctx.restore();

      const formattedB = b >= 0 ? `+ ${b.toFixed(2)}` : `- ${Math.abs(b).toFixed(2)}`;
      const eqText = `ŷ = ${m.toFixed(2)}x ${formattedB}`;
      
      ctx.font = 'bold 12px sans-serif';
      const eqMetrics = ctx.measureText(eqText);
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(margin + 8, margin + 8, eqMetrics.width + 16, 24, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#2563eb';
      ctx.textAlign = 'left';
      ctx.fillText(eqText, margin + 16, margin + 24);

      dataPoints.forEach((p) => {
        const pt = dataToCanvasCoords(p.x, p.y);
        
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      if (currentMode === 'predict' && predictedX !== null) {
        const predY = m * predictedX + b;
        const targetPt = dataToCanvasCoords(predictedX, predY);
        const bottomPt = dataToCanvasCoords(predictedX, 0);
        const leftPt = dataToCanvasCoords(0, predY);

        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(bottomPt.x, bottomPt.y);
        ctx.lineTo(targetPt.x, targetPt.y);
        ctx.lineTo(leftPt.x, targetPt.y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(targetPt.x, targetPt.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        const text = `X: ${predictedX.toFixed(1)} ➔ 예측 Y: ${predY.toFixed(1)}`;
        ctx.font = 'bold 12px sans-serif';
        const textWidth = ctx.measureText(text).width;
        
        let labelX = targetPt.x + 10;
        let labelY = targetPt.y - 12;

        if (labelX + textWidth > w - margin) labelX = targetPt.x - textWidth - 15;
        if (labelY < margin + 15) labelY = targetPt.y + 25;

        ctx.fillStyle = '#1e293b';
        ctx.beginPath();
        ctx.roundRect(labelX - 6, labelY - 14, textWidth + 12, 22, 6);
        ctx.fill();

        ctx.fillStyle = '#f8fafc';
        ctx.fillText(text, labelX, labelY);
      }

      updateMetricsDisplay();
    }

    function updateMetricsDisplay() {
      const { mse, r2 } = calculateMetrics();

      document.getElementById('metricMSE').innerText = (mse !== null) ? mse.toFixed(2) : '-';
      document.getElementById('metricSlope').innerText = m.toFixed(3);
      document.getElementById('metricIntercept').innerText = b.toFixed(3);
      
      const sign = b >= 0 ? '+' : '-';
      const absB = Math.abs(b).toFixed(3);
      const eqStr = `ŷ = ${m.toFixed(3)}x ${sign} ${absB}`;
      const eqElem = document.getElementById('equationText');
      if (eqElem) {
        eqElem.innerText = eqStr;
      }

      const r2Elem = document.getElementById('metricR2');
      if (r2 !== null) {
        r2Elem.innerText = r2.toFixed(3);
        if (r2 >= 0.7) r2Elem.className = "text-xl md:text-2xl font-black text-emerald-600";
        else if (r2 >= 0.3) r2Elem.className = "text-xl md:text-2xl font-black text-amber-600";
        else r2Elem.className = "text-xl md:text-2xl font-black text-rose-600";
      } else {
        r2Elem.innerText = '-';
        r2Elem.className = "text-xl md:text-2xl font-black text-emerald-600";
      }
    }

    function handleCanvasInteraction(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;

      const px = clientX - rect.left;
      const py = clientY - rect.top;

      const dataPt = canvasToDataCoords(px, py);

      if (currentMode === 'add') {
        dataPoints.push(dataPt);
        guideText.innerText = `📌 점이 추가되었습니다 (총 ${dataPoints.length}개). '학습 시작'을 눌러 최적의 회귀선을 찾아보세요.`;
      } else if (currentMode === 'predict') {
        predictedX = dataPt.x;
        const predY = m * predictedX + b;
        guideText.innerText = `🎯 X=${predictedX.toFixed(1)} 일 때, 현재 모델의 예측 값은 Y=${predY.toFixed(1)} 입니다.`;
      }

      render();
    }

    canvas.addEventListener('pointerdown', handleCanvasInteraction);

    epochsSlider.addEventListener('input', (e) => {
      epochsVal.innerText = `${e.target.value}회`;
    });

    lrSlider.addEventListener('input', (e) => {
      lrVal.innerText = parseFloat(e.target.value).toFixed(4);
    });

    function setMode(mode) {
      currentMode = mode;
      const addBtn = document.getElementById('modeAddBtn');
      const predictBtn = document.getElementById('modePredictBtn');

      if (mode === 'add') {
        addBtn.className = "flex-1 md:flex-none px-4 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 shadow-sm bg-white text-blue-600";
        predictBtn.className = "flex-1 md:flex-none px-4 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 text-slate-600 hover:text-slate-900";
        canvas.style.cursor = 'crosshair';
        guideBanner.className = "px-4 py-2.5 bg-amber-50 border-b border-amber-200 text-amber-800 text-xs sm:text-sm font-medium flex items-center gap-2";
        guideText.innerText = "📍 데이터 추가 모드: 캔버스를 클릭하여 데이터 점을 생성하세요.";
      } else {
        predictBtn.className = "flex-1 md:flex-none px-4 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 shadow-sm bg-white text-amber-600";
        addBtn.className = "flex-1 md:flex-none px-4 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 text-slate-600 hover:text-slate-900";
        canvas.style.cursor = 'pointer';
        guideBanner.className = "px-4 py-2.5 bg-amber-50 border-b border-amber-200 text-amber-800 text-xs sm:text-sm font-medium flex items-center gap-2";
        guideText.innerText = "🎯 예측하기 모드: 캔버스 위의 특정 X 위치를 클릭하여 예측 Y값을 확인해보세요.";
      }
      render();
    }

    function loadSampleData() {
      resetData();
      
      const samples = [
        { x: 15, y: 25 }, { x: 25, y: 32 }, { x: 35, y: 48 },
        { x: 45, y: 52 }, { x: 55, y: 68 }, { x: 65, y: 70 },
        { x: 75, y: 82 }, { x: 85, y: 88 }
      ];

      dataPoints = samples.map(p => ({
        x: p.x,
        y: Math.max(5, Math.min(95, p.y + (Math.random() * 12 - 6)))
      }));

      guideBanner.className = "px-4 py-2.5 bg-blue-50 border-b border-blue-200 text-blue-800 text-xs sm:text-sm font-medium flex items-center gap-2";
      guideText.innerText = "🎲 예시 데이터가 생성되었습니다. '▶ 학습 시작' 버튼을 눌러보세요!";
      render();
    }

    function resetData() {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      isTraining = false;
      dataPoints = [];
      m = 0;
      b = 50;
      predictedX = null;
      
      trainBtn.disabled = false;
      trainBtn.classList.remove('opacity-50', 'cursor-not-allowed');

      guideBanner.className = "px-4 py-2.5 bg-amber-50 border-b border-amber-200 text-amber-800 text-xs sm:text-sm font-medium flex items-center gap-2";
      guideText.innerText = "💡 캔버스를 터치하거나 클릭하여 데이터 점을 추가해보세요!";
      render();
    }

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('load', () => {
      resizeCanvas();
      loadSampleData();
    });
  </script>
</body>
</html>
"""

# Streamlit에 HTML 구성 요소 연동 (높이는 컨텐츠 크기에 맞춰 1050px 설정)
components.html(html_code, height=1050, scrolling=True)
