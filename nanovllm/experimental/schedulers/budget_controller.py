class PrefillBudgetController:
    """
    动态 Prefill Token Budget 控制器：
    通过监测 Decode 阶段的 ITL（Inter-Token Latency），动态调整 Scheduler 每轮允许执行的
    Prefill Token 数量，从而平衡 Prefill 吞吐量和 Decode 延迟。

    当前控制器只计算预算，不直接修改 Scheduler。
    """

    def __init__(
        self,
        min_budget: int,
        max_budget: int,
        initial_budget: int, #初始 Prefill Budget
        target_itl_ms: float,
        ema_alpha: float = 0.2, #EMA平滑系数
    ):
        if not 1 <= min_budget <= initial_budget <= max_budget:
            raise ValueError("budgets must satisfy 1 <= min <= initial <= max")
        if target_itl_ms <= 0:
            raise ValueError("target_itl_ms must be positive")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")

        self.min_budget = min_budget
        self.max_budget = max_budget
        self.current_budget = initial_budget

        self.target_itl_ms = target_itl_ms
        self.ema_alpha = ema_alpha
        self.ema_itl_ms = None      #经过 EMA 平滑后的 ITL

    def observe_itl(self, itl_ms: float, has_waiting_prefill: bool) -> int:
        """
        根据一次 Token 间隔观测，返回下一次使用的 Prefill Budget。

        itl_ms：相邻两个输出 Token 的时间间隔，单位是毫秒。
        has_waiting_prefill：当前是否有 Prefill 请求等待处理。
        """
        if itl_ms <= 0:
            raise ValueError("itl_ms must be positive")

        if self.ema_itl_ms is None:
            self.ema_itl_ms = itl_ms
        else:
            self.ema_itl_ms = (
                self.ema_alpha * itl_ms
                + (1 - self.ema_alpha) * self.ema_itl_ms        #EMA_t = alpha * ITL_t + (1-alpha) * EMA_{t-1}

            )

        # 没有 Prefill 等待时，暂时保持预算
        if not has_waiting_prefill:
            return self.current_budget

        """
        不直接使用每次测量的 ITL，而是使用 EMA

        因为 ITL 会存在波动
        如果每次测量之后都直接调整 Budget，可能导致 Budget 频繁变化,因此这里引入 EMA（指数移动平均）来平滑 ITL
        """

        if self.ema_itl_ms > self.target_itl_ms:
            # Decode 间隔偏大，收缩 Prefill 预算
            self.current_budget = max(
                self.min_budget,
                int(self.current_budget * 0.8),
            )
        elif self.ema_itl_ms < self.target_itl_ms * 0.8:
            # Decode 间隔较小，逐步增加 Prefill 预算
            self.current_budget = min(
                self.max_budget,
                self.current_budget + 32,
            )

        return self.current_budget
