# ref : https://github.com/quantylab/rltrader/blob/master/src/quantylab/rltrader/environment.py
## environment.py : 투자할 종목의 차트 관리
class Environment:
    PRICE_IDX = 0  # 종가의 위치

    def __init__(self, chart_data=None):
        # 차트 데이터
        self.chart_data = chart_data
        # 현재 관측치
        self.observation = None
        # 차트데이터에서 현재 위치
        self.idx = -1

    def reset(self):
        # idx와 observation 초기화
        self.observation = None
        self.idx = -1

    def observe(self):
        # idx를 다음 위치로 이동하고 observation을 업데이트
        # 함수에서 하루 앞으로 이동하며 차트 데이터에서 관측 데이터 제공, 더이상 제공 데이터가 없으면 none
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    # observation에서 종가 획득 후 반환
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None