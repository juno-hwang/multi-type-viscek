# Multi-type Vicsek Model Simulation

이 프로젝트는 Python (Taichi) 및 WebGL 시각화를 사용하여 여러 입자 유형을 갖는 Vicsek 모델을 시뮬레이션합니다.

## 모델 설명

이 시뮬레이션은 $M$개의 고유한 입자 유형을 포함하는 일반화된 Vicsek 모델을 구현합니다.

- **매개변수:**
    - $M$: 입자 type의 수.
    - $N_i$: i-type 입자의 수. 총 입자 수 $N = \sum_{i=1}^{M} N_i$.
    - $L$: 정사각형 시뮬레이션 영역의 크기 (주기적 경계 조건).
    - $R$: 상호작용 반경.
    - $\eta$: 노이즈 진폭.
    - $W_{ij}$: j-type 입자에서 i-type 입자로의 상호작용 강도 가중치.

- **상태 변수:**
    - $x_{ni}(t)$: 시간 $t$에서 i-type의 $n$번째 입자 위치.
    - $\theta_{ni}(t)$: 시간 $t$에서 i-type의 $n$번째 입자 속도 벡터의 각도.
    - $v_{ni}(t) = (\cos \theta_{ni}(t), \sin \theta_{ni}(t))$: 속도 벡터 (정규화됨).

- **업데이트 규칙:**
    1.  **각도 업데이트:** 다음 시간 단계 $t+1$에서 i-type의 $n$번째 입자의 각도는 반경 $R$ 내 이웃 입자들의 평균 각도(상호작용 행렬 $W$로 가중됨)에 무작위 노이즈 항을 더하여 결정됩니다.
        - 이웃 방향의 가중 합 계산:

          $$ S_{ni} = \sum_{j=1}^{M} \sum_{m=1}^{N_j} W_{ij} \sin(\theta_{mj}(t)) \cdot I(|x_{mj}(t) - x_{ni}(t)| \le R) $$
          
          $$ C_{ni} = \sum_{j=1}^{M} \sum_{m=1}^{N_j} W_{ij} \cos(\theta_{mj}(t)) \cdot I(|x_{mj}(t) - x_{ni}(t)| \le R) $$
          
        - $ \langle \theta_{ni}(t) \rangle = \operatorname{atan2}(S_{ni}, C_{ni}) $
          
        - 노이즈 추가:
          $$ \theta_{ni}(t+1) = \langle \theta_{ni}(t) \rangle + \Delta \theta_t $$
          여기서 $\Delta \theta_t$는 $[-\eta, \eta]$ 범위의 균일 분포 난수입니다.
    2.  **위치 업데이트:** 입자는 새로운 각도 방향으로 일정한 속력 $v$로 이동합니다.
        $$ x_{ni}(t+1) = (x_{ni}(t) + v_{ni}(t+1)) \pmod{L} $$
        모듈로 연산은 주기적 경계 조건을 보장합니다.

- **Order Parameter:** 
    $ \phi_i = \frac{1}{N_i} \left| \sum_{n=1}^{N_i} e^{i \theta_{ni}} \right| $

## WebGL 시뮬레이션 (`simulate.html`)

`simulate.html`은 다중 유형 Vicsek 모델의 상호작용 가능한 브라우저 기반 시각화를 제공합니다. JavaScript와 WebGL 렌더링을 사용하여 동일한 핵심 로직을 구현하며, Python이나 Taichi 없이도 실시간 매개변수 조정 및 시각화가 가능합니다.

## Python 시뮬레이션 (`vicsek_multi.py`)

이 스크립트는 [Taichi](https://www.taichi-lang.org/)를 사용하여 GPU에서 시뮬레이션을 가속화합니다.

**사용법:**

1.  **Taichi 설치:** 먼저 `taichi` 라이브러리를 설치해야 합니다:
    ```bash
    pip install taichi
    ```
2.  `if __name__ == "__main__":` 블록 내의 매개변수 수정:
    - `particle_counts_per_type` (`Ns`): 각 유형별 입자 수를 포함하는 리스트 (예: `[1000, 500]`)
    - `interaction_matrix` (`W`): $W_{ij}$ 행렬을 나타내는 NumPy 배열. 코드는 내부적으로 `W.T`를 사용하므로, $W_{ij}$가 유형 `j`에서 유형 `i`로의 영향을 나타내는 것을 기준으로 행렬을 정의해야 합니다.
    - `noise_level` (`eta`): 노이즈 진폭 $\eta$.
    - `L`, `v`, `R`, `warm_up_steps`, `steps_per_frame`과 같은 다른 매개변수는 `simulate_multiple` 함수 정의 또는 호출 부분에서 조정할 수 있습니다.
3.  스크립트 실행:
    ```bash
    python vicsek_multi.py
    ```

스크립트는 입자를 초기화하고, 준비 단계(선택 사항)를 실행한 다음, 시뮬레이션과 각 유형별 질서 매개변수를 표시하는 Taichi GUI 창을 엽니다.
