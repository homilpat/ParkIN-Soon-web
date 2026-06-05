import re

with open('kiosk.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Remove Step 1 form
p1 = r'    with st\.form\("olf_form", clear_on_submit=False\):\n(.*?)        submit_btn = st\.form_submit_button\("다음 단계 →  배변 상태 체크", use_container_width=True, type="primary"\)\n\n    if submit_btn:\n'
def repl1(m):
    inner = m.group(1).replace('\n        ', '\n    ')
    if inner.startswith('        '): inner = inner[4:]
    return inner + '    if st.button("다음 단계 →  배변 상태 체크", use_container_width=True, type="primary"):\n'
code = re.sub(p1, repl1, code, flags=re.DOTALL)

# Remove Step 2 form
p2 = r'    with st\.form\("constipation_form", clear_on_submit=False\):\n(.*?)        submit_btn = st\.form_submit_button\("다음 단계 →  나선 그리기", use_container_width=True, type="primary"\)\n\n    if submit_btn:\n'
def repl2(m):
    inner = m.group(1).replace('\n        ', '\n    ')
    if inner.startswith('        '): inner = inner[4:]
    return inner + '    if st.button("다음 단계 →  나선 그리기", use_container_width=True, type="primary"):\n'
code = re.sub(p2, repl2, code, flags=re.DOTALL)

with open('kiosk.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Removed form successfully")
