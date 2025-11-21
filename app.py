import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import nbinom, binom, norm
from openai import OpenAI
import json
import datetime

# ==========================================
# [보안 설정] Streamlit Cloud의 비밀키 가져오기
# ==========================================
try:
    # Streamlit Cloud에 등록된 키를 가져옵니다.
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # 로컬 등에서 키가 없을 경우 안내 메시지
    st.error("🚨 OpenAI API 키를 찾을 수 없습니다. Streamlit 설정에서 Secrets를 등록해주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==========================================
# 1. 품질 비용 계산 로직
# ==========================================
# 기준 파라미터
REF_PARAMS = {'lambda': 0.035, 'alpha': 4.2, 'mu': 2.11004, 'sigma': 0.78286}
A, CUTOFF, USL_FLAT = 706.9, 71, 3.5
LOT_SIZE, N_SAMPLE = 25, 5
ALPHA_TEST, BETA_TEST = 0.01, 0.02
COSTS = {'opp': 2500, 'scrap': 900, 'escape': 17100, 'inspect': 30}

def calculate_cost(lambda_d, alpha_d, mu_f, sigma_f):
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

REF_COST = calculate_cost(REF_PARAMS['lambda'], REF_PARAMS['alpha'], REF_PARAMS['mu'], REF_PARAMS['sigma'])

# ==========================================
# 2. LLM 파라미터 추출기
# ==========================================
def extract_params_from_text(user_text, current_params):
    system_prompt = f"""
    You are a data extraction assistant. 
    Extract parameters from user input and return a JSON object.
    - lambda (range 0.01-0.1)
    - alpha (range 1-10)
    - mu (range 1.2-3.0)
    - sigma (range 0.3-1.3)
    
    Current values: {current_params}
    Rules: 
    1. Update only mentioned parameters. Keep others same.
    2. Return JSON ONLY. keys: "lambda", "alpha", "mu", "sigma".
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ==========================================
# 3. 화면 구성 (UI)
# ==========================================
st.set_page_config(page_title="웨이퍼 단가 계산기", layout="wide")

# 초기값 설정
if 'params' not in st.session_state: st.session_state['params'] = REF_PARAMS.copy()
if 'history' not in st.session_state: st.session_state['history'] = []

# 사이드바
with st.sidebar:
    st.header("📜 질문 기록")
    for h in reversed(st.session_state['history'][-5:]):
        st.text(f"[{h['time']}]\n{h['query'][:15]}...")
    if st.button("초기화"):
        st.session_state['history'] = []
        st.session_state['params'] = REF_PARAMS.copy()
        st.rerun()

# 메인 화면
st.title("🤖 AI 웨이퍼 단가 계산기")
st.info("예: '알파 5.0, 람다 0.02일 때 공급사 Tier에 따른 단가표 보여줘'")

user_input = st.text_area("질문을 입력하세요", height=80)
calc_btn = st.button("계산하기")

if calc_btn and user_input:
    with st.spinner("AI가 분석 중입니다..."):
        try:
            new_params = extract_params_from_text(user_input, st.session_state['params'])
            st.session_state['params'] = new_params
            st.session_state['history'].append({"time": datetime.datetime.now().strftime("%H:%M"), "query": user_input})
        except Exception as e:
            st.error(f"오류 발생: {e}")

# 결과 계산
curr = st.session_state['params']
curr_cost = calculate_cost(curr['lambda'], curr['alpha'], curr['mu'], curr['sigma'])
diff = curr_cost - REF_COST
diff_pct = (diff / REF_COST) * 100

st.write("---")
st.markdown(f"**👉 현재 적용 파라미터:** `λ={curr['lambda']}`, `α={curr['alpha']}`, `μ={curr['mu']}`, `σ={curr['sigma']}`")

# 메트릭 표시
c1, c2, c3 = st.columns(3)
c1.metric("기준 비용", f"${REF_COST:,.0f}")
c2.metric("신규 비용", f"${curr_cost:,.0f}", delta=f"{diff:,.0f}", delta_color="inverse")
c3.metric("비용 변동률", f"{diff_pct:+.2f}%")

# 단가표 생성
tiers = [
    {'Tier': 'Tier 1', 'k': 0.3, 'Desc': '전략 파트너'},
    {'Tier': 'Tier 2', 'k': 0.5, 'Desc': '주요 공급사'},
    {'Tier': 'Tier 3', 'k': 0.7, 'Desc': '일반 공급사'}
]

data = []
for t in tiers:
    adj = -t['k'] * diff_pct
    direction = "인상 (▲)" if adj > 0 else "인하 (▼)"
    if abs(adj) < 0.01: direction = "-"
    data.append([t['Tier'], t['k'], f"{adj:+.2f}% {direction}", t['Desc']])

st.subheader("💰 단가 조정 가이드라인")
st.table(pd.DataFrame(data, columns=['Tier', '협상계수(k)', '조정률', '비고']))