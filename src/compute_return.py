def add_return_column(data):
    # 按基金代码独立计算净值收益率，避免不同基金之间串扰
    data["return"] = data.groupby("code")["nav"].pct_change()
    return data
