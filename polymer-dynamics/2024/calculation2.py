import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

def c_input_of(t, t_0, c_0) : 
    # t_0 means width of Input function
    # c_0 means height of peak 
    return c_0 * (np.heaviside(t, 1) - np.heaviside(t-t_0, 1))

def p_n_of(t, n):
    t_n = t_n_arr[n]
    return np.where(
        (t_n - t_w <= t) & (t < t_n),
        (t - t_n) / t_w + 1,
        np.where(
            (t_n <= t) & (t <= t_n + t_w),
            1 - (t - t_n) / t_w,
            0
        )
    )

def R_of(t, N):
    sum = np.zeros_like(t)
    for n in np.arange(1, N):
        sum += r_n[n] * p_n_of(t, n)
    return sum

def c_out_of(t, N):
    sum = np.zeros_like(t)
    for n in np.arange(1, N):
        p_n_of_t = p_n_of(t, n)
        q_n_of_t = np.convolve(p_n_of_t, c_input, 'same')
        sum += r_n[n] * q_n_of_t
    return sum


# 구간 적분 함수 정의
def integral_q_n(n, a, b):
    integrand = lambda t: q_n[n-1][t_index(t)]
    result, error = quad(integrand, a, b)
    return result

# t를 인덱스로 변환하는 함수
def t_index(t_val):
    return int((t_val - t[0]) / (t[1] - t[0]))

t_min = 0   
t_max = 50
t_out = 50
t_w = 2
N = 25
N_arr = np.arange(1, N+1)

t_n_arr = np.arange(1, N+1) * t_w
r_n = np.ones_like(t_n_arr)

t = np.arange(-100, 100, 0.1)
R_of_t = R_of(t, N)
c_input = c_input_of(t, 1, 1)
c_output = c_out_of(t, N)

plt.plot(t, c_input, label="input function")
plt.plot(t, c_output, label="output function")
plt.legend()
plt.show()

p_n = []
q_n = []
for i in range(1, N+1) :
    p_n_of_t = p_n_of(t, i-1)
    p_n.append(p_n_of_t)
    q_n_of_t = np.convolve(p_n_of_t, c_input, 'same')
    q_n.append(q_n_of_t)

    #plt.plot(t, p_n_of_t)
    #plt.plot(t, q_n_of_t, label=f'q_{i}(t)')
    #plt.legend()
#plt.show()
# p_n_of_t = p_n_of(t, 1)
# q_n_of_t = np.convolve(p_n_of_t, c_input, 'same')

# S_nm 행렬 초기화
S_nm = np.zeros((N, N))

a, b = t_min, t_max  # 적분 구간 설정
for i in range(1, N + 1):
    for j in range(1, N+1):
        integral_result = integral_q_n(i, a, b)
        integral_result_2 = integral_q_n(j, a, b)
        S_nm[i-1, j-1] = integral_result * integral_result_2

# S_nm 행렬을 DataFrame으로 변환
df = pd.DataFrame(S_nm, index=np.arange(1, N+1), columns=np.arange(1, N+1))

# DataFrame을 Excel 파일로 저장
df.to_excel("S_nm_matrix.xlsx", index=True)

# DataFrame을 텍스트 파일로 저장
df.to_csv("S_nm_matrix.txt", sep='\t', index=True)


