import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import nbinom, binom, norm
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
# 1. 품질 비용 계산 로직 (Math Backend)
# ==========================================
# 기준 파라미터 (Baseline)
REF_PARAMS = {
    'lambda': 0.035, 'alpha': 4.2,
    'mu': 2.11004, 'sigma': 0.78286
}
A, CUTOFF, USL_FLAT = 706.9, 71, 3.5
LOT_SIZE, N_SAMPLE = 25, 5
ALPHA_TEST, BETA_TEST = 0.01, 0.02
COSTS = {'opp': 2500, 'scrap': 900, 'escape': 17100, 'inspect': 30}

def calculate_cost(lambda_d, alpha_d, mu_f, sigma_f):
    """로트당 총 품질 비용 계산 (일반화 모델)"""
    # 1. 결함 불량 확률
    mu_val = lambda_d * A
    p_nb = alpha_d / (alpha_d + mu_val)
    p_defect = 1 - nbinom.cdf(CUTOFF - 1, alpha_d, p_nb)

    # 2. 평탄도 불량 확률
    p_flat = 1 - norm.cdf(USL_FLAT, loc=mu_f, scale=sigma_f)

    # 3. 통합 불량 확률
    p_total = 1 - (1 - p_defect) * (1 - p_flat)

    # 4. 검사 판정
    p_prime = (1 - p_total) * ALPHA_TEST + p_total * (1 - BETA_TEST)
    P_accept = binom.cdf(0, N_SAMPLE, p_prime)
    P_reject = 1 - P_accept

    # 5. 비용 합산
    cost = (
        P_reject * (1 - p_total) * LOT_SIZE * COSTS['opp'] +
        P_reject * p_total * LOT_SIZE * COSTS['scrap'] +
        P_accept * (p_total * LOT_SIZE) * COSTS['escape'] +
        N_SAMPLE * COSTS['inspect']
    )
    return cost

# 기준 비용 계산
REF_COST = calculate_cost(REF_PARAMS['lambda'], REF_PARAMS['alpha'], REF_PARAMS['mu'], REF_PARAMS['sigma'])

# ==========================================
# 2. LLM 파라미터 추출기 (개선됨)
# ==========================================
def extract_params_from_text(user_text):
    """
    사용자 입력에서 변경된 파라미터만 추출.
    언급되지 않은 파라미터는 null로 반환하도록 유도.
    """
    system_prompt = """
    You are a parameter extraction assistant.
    Extract the following parameters from the user's input:
    - "lambda" (Defect Density)
    - "alpha" (Cluster Parameter)
    - "mu" (Mean TTV)
    - "sigma" (Std Dev TTV)

    Rules:
    1. Extract ONLY the values explicitly mentioned by the user.
    2. If a parameter is NOT mentioned, set its value to null.
    3. Do NOT infer or guess values from context like "standard" or "baseline". Just return null.
    4. Return a JSON object. Example: {"lambda": 0.05, "alpha": null, "mu": null, "sigma": 0.78}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"},
            temperature=0  # 환각 방지를 위해 0으로 설정
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"AI 추출 오류: {e}")
        return {}

# ==========================================
# 3. Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="웨이퍼 단가 계산기", layout="wide")

# 세션 상태 초기화
if 'params' not in st.session_state:
    st.session_state['params'] = REF_PARAMS.copy()
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 사이드바
with st.sidebar:
    st.header("📜 질문 기록")
    if st.button("초기화 (Reset)"):
        st.session_state['history'] = []
        st.session_state['params'] = REF_PARAMS.copy()
        st.rerun()
    
    for i, h in enumerate(reversed(st.session_state['history'])):
        st.text(f"[{h['time']}] {h['query'][:15]}...")
        if i > 4: break

# 메인 화면
st.title("🤖 AI 웨이퍼 단가 계산기")
st.info("💡 예시: '알파는 5.0, 람다는 0.02로 설정해줘' (나머지는 기존 값 유지)")

# 입력창
user_input = st.text_area("질문을 입력하세요", height=80)
calc_btn = st.button("계산하기")

if calc_btn and user_input:
    with st.spinner("AI가 파라미터를 분석 중입니다..."):
        # 1. AI에게 추출 요청 (변경된 것만 null 아닌 값으로 옴)
        extracted = extract_params_from_text(user_input)
        
        # 2. 기존 파라미터에 덮어쓰기 (Merge)
        current_params = st.session_state['params']
        updated_params = current_params.copy()
        
        changes = []
        for key, val in extracted.items():
            if val is not None:
                updated_params[key] = float(val) # 숫자 변환 확인
                changes.append(f"{key}: {val}")
        
        # 3. 상태 업데이트
        st.session_state['params'] = updated_params
        
        # 4. 기록
        st.session_state['history'].append({
            "time": datetime.datetime.now().strftime("%H:%M"),
            "query": user_input,
            "changes": ", ".join(changes) if changes else "변경 없음"
        })
        
        if not changes:
            st.warning("⚠️ 변경된 파라미터를 찾지 못했습니다. (기존 값 유지)")
        else:
            st.success(f"✅ 파라미터 업데이트: {', '.join(changes)}")

# 결과 계산 및 표시
curr = st.session_state['params']
curr_cost = calculate_cost(curr['lambda'], curr['alpha'], curr['mu'], curr['sigma'])
diff = curr_cost - REF_COST
diff_pct = (diff / REF_COST) * 100

st.write("---")
# 현재 파라미터 명시적 표시
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("Lambda (λ)", f"{curr['lambda']}")
col_p2.metric("Alpha (α)", f"{curr['alpha']}")
col_p3.metric("Mean (μ)", f"{curr['mu']}")
col_p4.metric("Sigma (σ)", f"{curr['sigma']}")

# 비용 결과 표시
st.subheader("📊 비용 분석 결과")
c1, c2, c3 = st.columns(3)
c1.metric("기준 비용 (Baseline)", f"${REF_COST:,.2f}")
c2.metric("신규 비용 (Current)", f"${curr_cost:,.2f}", delta=f"{diff:,.2f}", delta_color="inverse")
c3.metric("비용 증감율", f"{diff_pct:+.2f}%")

# 단가표 생성
tiers = [
    {'Tier': 'Tier 1', 'k': 0.3, 'Desc': '전략 파트너'},
    {'Tier': 'Tier 2', 'k': 0.5, 'Desc': '주요 공급사'},
    {'Tier': 'Tier 3', 'k': 0.7, 'Desc': '일반 공급사'}
]

data = []
for t in tiers:
    # 단가 조정 공식: -k * (비용증감율)
    adj = -t['k'] * diff_pct
    
    direction = "변동 없음"
    if adj > 0.001: direction = "인상 (▲)"
    elif adj < -0.001: direction = "인하 (▼)"
    
    data.append([t['Tier'], t['k'], f"{adj:+.2f}% {direction}", t['Desc']])

st.subheader("💰 단가 조정 가이드라인")
st.table(pd.DataFrame(data, columns=['Tier', '협상계수(k)', '조정률', '비고']))