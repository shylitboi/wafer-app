import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, binom, norm
from mpl_toolkits.mplot3d import Axes3D
from openai import OpenAI
import json
import datetime

# ==========================================
# [설정] API 키 보안 처리
# ==========================================
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    st.error("🚨 OpenAI API 키가 없습니다. Secrets에 등록해주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==========================================
# 1. 고정된 기준 파라미터 (Immutable Baseline)
# ==========================================
# 사용자가 무슨 값을 입력하든, 비교의 기준은 항상 이 값들입니다.
REF_PARAMS = {
    'lambda': 0.035,  # 결함 밀도
    'alpha': 4.2,     # 클러스터링
    'mu': 2.11004,    # 평탄도 평균
    'sigma': 0.78286  # 평탄도 산포
}

# 공통 상수
A, CUTOFF, USL_FLAT = 706.9, 71, 3.5
LOT_SIZE, N_SAMPLE = 25, 5
ALPHA_TEST, BETA_TEST = 0.01, 0.02
COSTS = {'opp': 2500, 'scrap': 900, 'escape': 17100, 'inspect': 30}

# 비용 계산 함수
def calculate_total_cost(lambda_d, alpha_d, mu_f, sigma_f):
    """4개 변수를 받아 로트당 총 비용을 계산"""
    # 1. 확률 계산
    mu_val = lambda_d * A
    p_nb = alpha_d / (alpha_d + mu_val)
    p_defect = 1 - nbinom.cdf(CUTOFF - 1, alpha_d, p_nb)
    p_flat = 1 - norm.cdf(USL_FLAT, loc=mu_f, scale=sigma_f)
    p_total = 1 - (1 - p_defect) * (1 - p_flat)

    # 2. 검사 판정
    p_prime = (1 - p_total) * ALPHA_TEST + p_total * (1 - BETA_TEST)
    P_accept = binom.cdf(0, N_SAMPLE, p_prime)
    P_reject = 1 - P_accept

    # 3. 비용 합산 (일반화 모델)
    cost = (
        P_reject * (1 - p_total) * LOT_SIZE * COSTS['opp'] +
        P_reject * p_total * LOT_SIZE * COSTS['scrap'] +
        P_accept * (p_total * LOT_SIZE) * COSTS['escape'] +
        N_SAMPLE * COSTS['inspect']
    )
    return cost

# 기준 비용 미리 계산 (고정값)
COST_REF = calculate_total_cost(REF_PARAMS['lambda'], REF_PARAMS['alpha'], REF_PARAMS['mu'], REF_PARAMS['sigma'])

# ==========================================
# 2. LLM 파라미터 추출기 (항상 기준값 베이스)
# ==========================================
def extract_params_from_text(user_text):
    """
    사용자 입력에서 파라미터를 추출하되, 
    언급되지 않은 값은 무조건 '기준 파라미터(REF_PARAMS)'를 따름.
    """
    system_prompt = f"""
    You are a data extraction assistant.
    
    [Baseline Parameters]
    - lambda: {REF_PARAMS['lambda']}
    - alpha: {REF_PARAMS['alpha']}
    - mu: {REF_PARAMS['mu']}
    - sigma: {REF_PARAMS['sigma']}

    Rules:
    1. Extract values from the user's input.
    2. If a parameter is mentioned, use the user's value.
    3. If a parameter is NOT mentioned, use the [Baseline Parameters] value above. (DO NOT use previous context)
    4. Return a JSON object with keys: "lambda", "alpha", "mu", "sigma".
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)

# ==========================================
# 3. Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="웨이퍼 단가 계산기", layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_params' not in st.session_state:
    st.session_state['current_params'] = REF_PARAMS.copy()

# [사이드바]
with st.sidebar:
    st.header("📜 질문 기록")
    if st.button("기록 초기화"):
        st.session_state['history'] = []
        st.session_state['current_params'] = REF_PARAMS.copy()
        st.rerun()
    
    for i, h in enumerate(reversed(st.session_state['history'])):
        st.text(f"[{h['time']}] {h['query'][:15]}...")
        if i > 4: break

# [메인 화면]
st.title("🤖 웨이퍼 단가 계산기")
st.caption("made by 샤리보🧑🏻‍💻") 
st.markdown(f"""
**기준 파라미터 (Fixed Baseline):** `λ={REF_PARAMS['lambda']}`, `α={REF_PARAMS['alpha']}`, `μ={REF_PARAMS['mu']}`, `σ={REF_PARAMS['sigma']}`  

""")

# 입력창
user_input = st.text_area("질문을 입력하세요 (예: '알파만 5.0으로 바꾸면?')", height=80)
calc_btn = st.button("계산하기")

if calc_btn and user_input:
    with st.spinner("AI가 기준값을 바탕으로 분석 중입니다..."):
        try:
            # 항상 REF_PARAMS를 기준으로 추출
            new_params = extract_params_from_text(user_input)
            st.session_state['current_params'] = new_params
            
            # 비용 계산
            new_cost = calculate_total_cost(
                new_params['lambda'], new_params['alpha'], new_params['mu'], new_params['sigma']
            )
            
            # 기록 저장
            st.session_state['history'].append({
                "time": datetime.datetime.now().strftime("%H:%M"),
                "query": user_input
            })
        except Exception as e:
            st.error(f"오류 발생: {e}")

# 현재 계산 결과
curr = st.session_state['current_params']
curr_cost = calculate_total_cost(curr['lambda'], curr['alpha'], curr['mu'], curr['sigma'])
diff_pct = ((curr_cost - COST_REF) / COST_REF) * 100

st.divider()
st.subheader("1️⃣ 비용 분석 및 단가표")

# 메트릭
c1, c2, c3, c4 = st.columns(4)
c1.metric("적용 λ (Lambda)", f"{curr['lambda']}")
c2.metric("적용 α (Alpha)", f"{curr['alpha']}")
c3.metric("적용 μ (Mean)", f"{curr['mu']}")
c4.metric("적용 σ (Sigma)", f"{curr['sigma']}")

m1, m2, m3 = st.columns(3)
m1.metric("기준 품질비용 (Fixed)", f"${COST_REF:,.2f}")
m2.metric("신규 품질비용 (Current)", f"${curr_cost:,.2f}", delta=f"{curr_cost - COST_REF:,.2f}", delta_color="inverse")
m3.metric("비용 변동률", f"{diff_pct:+.2f}%")

# 단가표
tiers = [
    {'Tier': 'Tier 1', 'k': 0.3, 'Desc': '전략 파트너'},
    {'Tier': 'Tier 2', 'k': 0.5, 'Desc': '주요 공급사'},
    {'Tier': 'Tier 3', 'k': 0.7, 'Desc': '일반 공급사'}
]
data = []
for t in tiers:
    adj = -t['k'] * diff_pct
    direction = "변동 없음"
    if adj > 0.001: direction = "인상 (▲)"
    elif adj < -0.001: direction = "인하 (▼)"
    data.append([t['Tier'], t['k'], f"{adj:+.2f}% {direction}", t['Desc']])

st.table(pd.DataFrame(data, columns=['Tier', '협상계수(k)', '조정률', '비고']))

# ==========================================
# 4. 4D Interactive Plot (Streamlit Slider 버전)
# ==========================================
st.divider()
st.subheader("2️⃣ 4D Interactive Visualization")
st.markdown("아래 슬라이더를 움직여 결함 균질도(α)와 평탄도 산포(σ)가 단가에 미치는 영향을 확인하세요.")

# 4D Plot용 데이터 그리드 (미리 생성)
l_vals = np.linspace(0.01, 0.10, 20)
m_vals = np.linspace(1.2, 3.0, 20)
L_3d, M_3d = np.meshgrid(l_vals, m_vals)

# 🔹 [New] 파라미터 슬라이더와 시각 각도 슬라이더를 분리하여 배치
col_param1, col_param2, col_param3 = st.columns(3)
s_alpha = col_param1.slider("Cluster Parameter (α)", 1.0, 10.0, 4.2, 0.1)
s_sigma = col_param2.slider("Flatness Sigma (σ)", 0.3, 1.25, 0.78, 0.05)
s_k = col_param3.slider("Negotiation Factor (k)", 0.1, 1.0, 0.5, 0.1)

st.caption("👀 **시각 각도 조절 (View Angle)**")
col_view1, col_view2 = st.columns(2)
view_azim = col_view1.slider("회전 (Azimuth)", 0, 360, 315, 5) # 기본값 315도
view_elev = col_view2.slider("높이 (Elevation)", 0, 90, 25, 5)   # 기본값 25도

# 3D Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Z축(단가 조정률) 계산
Z_3d = np.zeros_like(L_3d)
for i in range(L_3d.shape[0]):
    for j in range(L_3d.shape[1]):
        # 사용자가 슬라이더로 선택한 alpha, sigma 적용
        c = calculate_total_cost(L_3d[i,j], s_alpha, M_3d[i,j], s_sigma)
        # 단가 조정률
        Z_3d[i,j] = -s_k * ((c - COST_REF) / COST_REF) * 100

# 서피스 플롯
surf = ax.plot_surface(L_3d, M_3d, Z_3d, cmap='coolwarm', edgecolor='none', alpha=0.85, vmin=-100, vmax=20)

# 기준점 표시 (Baseline)
if np.isclose(s_alpha, REF_PARAMS['alpha'], atol=0.5) and np.isclose(s_sigma, REF_PARAMS['sigma'], atol=0.1):
    ax.scatter(REF_PARAMS['lambda'], REF_PARAMS['mu'], 0, color='yellow', s=200, marker='*', edgecolors='black', label='Baseline', zorder=10)
    ax.legend()

ax.set_xlabel('Defect Density λ')
ax.set_ylabel('Mean TTV μ')
ax.set_zlabel('ΔPrice (%)')
ax.set_title(f'Price Sensitivity Surface\n(α={s_alpha}, σ={s_sigma}, k={s_k})', fontsize=14)
ax.set_zlim(-100, 20)

# 🔹 [Changed] 사용자가 슬라이더로 조절한 각도 적용
ax.view_init(elev=view_elev, azim=view_azim)

fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label='Price Adj (%)')
st.pyplot(fig)