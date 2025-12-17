import unicodedata
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 테이블 도출할 때 간격 고려해서 출력하기
def text_width(s):
    s = str(s)
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w

def pad_text(s, width):
    s = str(s)
    return s + " " * max(0, width - text_width(s))

def print_table(df, title, w1=22, w2=20, w3=10):
    if df is None or df.empty:
        print("\n(추천 결과 없음)")
        return

    col1, col2, col3 = "학과", "분류", "적합도"

    line = "+" + "-"*(w1+2) + "+" + "-"*(w2+2) + "+" + "-"*(w3+2) + "+"
    print("\n=== {} ===".format(title))
    print(line)
    print("| {} | {} | {} |".format(pad_text(col1, w1), pad_text(col2, w2), pad_text(col3, w3)))
    print(line)

    for _, r in df.iterrows():
        print("| {} | {} | {} |".format(
            pad_text(r[col1], w1),
            pad_text(r[col2], w2),
            pad_text("{:.2f}".format(float(r[col3])), w3)
        ))
    print(line)

# 예,아니오 형태의 응답을 안정적으로 처리하여 오류를 방지하는 입력 함수
def input_int(msg, lo, hi):
    while True:
        try:
            v = int(input(msg).strip())
            if lo <= v <= hi:
                return v
            print(" {}~{} 사이 숫자만 입력해주세요.".format(lo, hi))
        except:
            print(" 숫자로 입력해주세요.")

def input_yes_no(msg):
    while True:
        s = input(msg).strip().lower()
        if s in ("y", "yes", "ㅇ", "ㅇㅇ", "네", "예"):
            return True
        if s in ("n", "no", "ㄴ", "ㄴㄴ", "아니", "아니요"):
            return False
        print("y/n 로 입력해주세요. (예: y 또는 n)")

# 전공 데이터 30개 중 선호도 1~10을 9개의 기준으로 나누기
전공데이터 = {
    "학과": [
        "컴퓨터공학과","소프트웨어학과","인공지능학과","정보보호학과","데이터사이언스학과",
        "전기전자공학과","기계공학과","산업공학과","신소재공학과","화학공학과",
        "건축학과","건축공학과","토목공학과","환경공학과","바이오의공학과",
        "수학과","통계학과","물리학과","화학과","생명과학과",
        "경영학과","회계학과","경제학과","국제통상학과","마케팅학과",
        "심리학과","사회학과","정치외교학과","미디어커뮤니케이션학과","디자인학과"
    ],
    "팀플선호":        [8,8,8,6,6, 6,6,8,6,6, 7,6,6,6,6, 2,4,4,4,4, 9,8,4,6,9, 6,6,6,9,6],
    "프로젝트선호":    [9,9,9,7,7, 7,7,8,6,6, 8,8,6,6,8, 2,6,4,4,4, 8,6,4,6,8, 6,4,4,9,9],
    "학습량":          [8,8,8,8,8, 9,9,7,9,9, 8,8,9,8,9, 7,7,8,8,8, 6,6,7,6,6, 6,6,6,5,7],
    "취업선호":        [9,9,9,8,9, 8,7,8,6,7, 6,6,6,6,7, 5,7,4,5,6, 8,8,7,7,8, 6,6,6,7,7],
    "발표선호":        [3,3,3,3,3, 3,3,4,3,3, 6,5,4,4,4, 3,3,3,3,3, 7,6,5,6,8, 6,6,8,9,8],
    "창의성":          [7,7,7,6,6, 6,6,6,6,6, 7,6,5,5,6, 6,5,5,5,5, 6,5,5,6,7, 6,5,5,9,9],
    "실습선호":        [9,9,9,8,8, 8,8,7,8,8, 8,8,7,7,8, 3,6,4,4,4, 4,3,3,4,5, 4,3,3,6,9],
    "진로다양성":      [9,9,9,8,9, 8,7,9,6,6, 6,6,6,6,7, 7,9,6,6,7, 9,8,8,8,8, 7,6,6,8,7],
    "사고력":          [9,9,9,8,8, 9,9,8,9,9, 8,8,8,8,8, 9,8,9,8,8, 7,7,8,7,7, 7,7,8,7,7]
}

df = pd.DataFrame(전공데이터)

# 30개의 학과를 분류하고, 이후 추천 결과에 사용할 분류 태그 정의
분류표 = {
    "컴퓨터공학과":"공학/IT","소프트웨어학과":"공학/IT","인공지능학과":"공학/IT","정보보호학과":"공학/IT","데이터사이언스학과":"공학/IT",
    "전기전자공학과":"공학","기계공학과":"공학","산업공학과":"공학","신소재공학과":"공학","화학공학과":"공학",
    "건축학과":"공학/설계","건축공학과":"공학","토목공학과":"공학","환경공학과":"공학","바이오의공학과":"공학/바이오",
    "수학과":"자연과학","통계학과":"자연과학","물리학과":"자연과학","화학과":"자연과학","생명과학과":"자연과학/바이오",
    "경영학과":"상경","회계학과":"상경","경제학과":"상경/사회","국제통상학과":"상경/사회","마케팅학과":"상경",
    "심리학과":"사회과학","사회학과":"사회과학","정치외교학과":"사회과학","미디어커뮤니케이션학과":"미디어/사회","디자인학과":"예술/디자인"
}

df["분류"] = df["학과"].map(분류표).fillna("기타")


기준들 = ["팀플선호","프로젝트선호","학습량","취업선호","발표선호","창의성","실습선호","진로다양성","사고력"]

질문문구 = {
    "팀플선호": "팀플 선호도",
    "프로젝트선호": "프로젝트 선호도",
    "학습량": "학습량(과제/공부량) 수용도",
    "취업선호": "취업 선호도",
    "발표선호": "발표 선호도",
    "창의성": "창의성 선호도",
    "실습선호": "실습 선호도",
    "진로다양성": "진로 다양성 선호도",
    "사고력": "사고력/문제해결 선호도"
}

# 모델 준비
scaler = StandardScaler()
X = scaler.fit_transform(df[기준들].astype(float).values)

knn = NearestNeighbors(n_neighbors=len(df), metric="euclidean")
knn.fit(X)

def calc_fitness(dist):
    return 100.0 / (1.0 + float(dist))

# 선호도 1~10 입력, 1과 10 기준 설정
def get_user_preferences():
    print("\n=== 성향 입력 (1~10) ===")
    print("1 = 매우 낮음 / 10 = 매우 높음\n")
    user = {}
    for k in 기준들:
        user[k] = input_int("{} (1~10): ".format(질문문구[k]), 1, 10)
    return user

# 사용자 선택 맞춤형 전공 5가지 추천
def recommend_major_top5(user, 제외, top_n=5):
    u_vec = np.array([[user[k] for k in 기준들]], dtype=float)
    u_scaled = scaler.transform(u_vec)

    dists, idx = knn.kneighbors(u_scaled)
    dists = dists[0]
    idx = idx[0]

    rows = []
    for d, i in zip(dists, idx):
        학과 = df.loc[i, "학과"]
        if 학과 in 제외:
            continue
        rows.append((학과, df.loc[i, "분류"], calc_fitness(d)))
        if len(rows) >= top_n:
            break

    if len(rows) == 0 and len(제외) > 0:
        제외.clear()
        return recommend_major_top5(user, 제외, top_n)

    return pd.DataFrame(rows, columns=["학과","분류","적합도"])

# 만족도 10이 되면 주전공 확정
def select_major(top5_df):
    print("\n=== 주전공 선택 (만족도 10점이면 확정됩니다) ===")
    for 학과 in top5_df["학과"].tolist():
        s = input_int("{} 만족도(1~10): ".format(학과), 1, 10)
        if s == 10:
            return 학과
    return None

# 복수전공 추천 학과를 1순위부터 5순위까지 도출하는 함수
def recommend_minor_top5(user, 주전공, top_k=5, 다른분류선호=True, 분류가점=0.15, 적합가중=1.0, 보완가중=0.55):
    u = scaler.transform(np.array([[user[k] for k in 기준들]], dtype=float))[0]
    Xall = scaler.transform(df[기준들].astype(float).values)

    main_idx = df.index[df["학과"] == 주전공][0]
    m = Xall[main_idx]
    main_cat = df.loc[main_idx, "분류"]

    results = []
    for i in range(len(df)):
        부전공 = df.loc[i, "학과"]
        if 부전공 == 주전공:
            continue

        a = Xall[i]
        dist_u = np.linalg.norm(a - u)
        fit = 1.0 / (1.0 + dist_u)

        dist_m = np.linalg.norm(a - m)
        comp = dist_m / (1.0 + dist_m)

        minor_cat = df.loc[i, "분류"]
        bonus = 분류가점 if (다른분류선호 and minor_cat != main_cat) else 0.0

        score = (적합가중 * fit) + (보완가중 * comp) + bonus
        results.append((부전공, minor_cat, score * 100.0))

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:top_k]
    return pd.DataFrame(top, columns=["학과","분류","적합도"])

# 확정 및 종료 함수 설정
def main():
    user = get_user_preferences()
    제외 = set()

    tries = 0
    MAX_TRIES = 100

    while True:
        tries += 1
        if tries > MAX_TRIES:
            print("\n추천 반복이 너무 길어져서 종료합니다.")
            return

        top5 = recommend_major_top5(user, 제외, 5)
        if top5.empty:
            print("\n추천 결과가 없습니다. 종료합니다.")
            return

        print_table(top5, "추천 전공 TOP5 (학과/분류/적합도)")

        주전공 = select_major(top5)
        if 주전공 is not None:
            print("\n 주전공이 확정되었습니다: {}".format(주전공))

            if input_yes_no("복수전공도 하시겠습니까? (y/n): "):
                minors = recommend_minor_top5(user, 주전공, 5, True)
                print_table(minors, "복수전공 추천 TOP5 (주전공: {})".format(주전공))
            else:
                print("\n복수전공은 하지 않으시는 것으로 선택되었습니다.")

            print("\n프로그램을 종료합니다.")
            return

        for x in top5["학과"].tolist():
            제외.add(x)

        if not input_yes_no("아직 주전공이 확정되지 않았습니다. 다른 추천을 보시겠습니까? (y/n): "):
            print("\n프로그램을 종료합니다.")
            return

if __name__ == "__main__":
    main()
