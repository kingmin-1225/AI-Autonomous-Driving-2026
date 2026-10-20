"""수집·추론 GUI 번역. 중국어는 간체를 사용한다."""
from terminal_text import tr

LANGUAGES = ('kor', 'eng', 'chn')


def normalize_language(value):
    if not isinstance(value, str) or value not in LANGUAGES:
        raise ValueError(tr('language는 kor, eng, chn 중 하나여야 합니다.'))
    return value


TEXT = {
    'ready': ('수집 대기 · Enter로 시작', 'Ready - Enter to record', '等待采集 · 按 Enter 开始'),
    'recording': ('녹화 중 · 저장하려면 1바퀴를 돌고 나서 TAB을 누르세요', 'RECORDING - Complete one lap, then press TAB to save', '正在录制 · 跑完一圈后按 TAB 保存'),
    'driving': ('모델 추론 중 · 자동 운전', 'INFERENCE RUNNING - automatic driving', '模型推理中 · 自动驾驶'),
    'finished': ('한 바퀴 완료 · 주행 정지', 'Lap complete - stopped', '已完成一圈 · 已停车'),
    'aborted': ('차선 이탈 5초 경과 · 주행 종료', 'Off lane for 5 seconds - stopped', '驶出车道达到5秒 · 驾驶结束'),
    'save_blocked': ('저장 불가: 한 바퀴 완료 후 TAB', 'Cannot save: complete a lap first', '无法保存：请先跑完一圈再按 TAB'),
    'saved': ('저장 완료', 'Saved successfully', '保存成功'),
    'discarded': ('폐기 완료 · 다시 시작', 'Discarded - restart', '已丢弃数据 · 重新开始'),
    'start': ('Enter 녹화 시작  |  ', 'Enter: Record  |  ', 'Enter 开始录制  |  '),
    'drive_keys': ('← → 조향  ↑ 가속  ↓ 제동', 'Left/Right: Steer  Up: Gas  Down: Brake', '← → 转向  ↑ 加速  ↓ 刹车'),
    'save_keys': ('TAB 저장 후 다시 시작', 'TAB: Save & restart', 'TAB 保存并重新开始'),
    'save_later_keys': ('TAB 한 바퀴 완료 후 저장', 'TAB: Save after a lap', 'TAB 跑完一圈后保存'),
    'exit_keys': ('  |  SPACE 폐기 후 다시 시작  |  ESC / 창 닫기 폐기 후 종료', '  |  SPACE: Discard & restart  |  ESC / Close: Discard & exit', '  |  SPACE 丢弃并重新开始  |  ESC / 关闭窗口 丢弃并退出'),
    'model': ('모델: ', 'Model: ', '模型：'),
    'benchmark_keys': ('고정 트랙 평가  |  ESC / 창 닫기 종료', 'Fixed-track evaluation  |  ESC / Close: Exit', '固定赛道评估  |  ESC / 关闭窗口 退出'),
    'infer_keys': ('ENTER 같은 트랙 다시 시작  |  ESC / 창 닫기 종료', 'ENTER: Restart same track  |  ESC / Close: Exit', 'ENTER 重新开始当前赛道  |  ESC / 关闭窗口 退出'),
    'collect_title': ('CarRacing | 데이터 수집', 'CarRacing | Data collection', 'CarRacing | 数据采集'),
    'infer_title': ('CarRacing | 모델 추론', 'CarRacing | Model inference', 'CarRacing | 模型推理'),
    'unsaved': (' · 미저장 · TAB으로 저장 후 다시 시작', ' - Unsaved - TAB: Save & restart', ' · 未保存 · TAB保存并重新开始'),
    'metric_header': ('점수 항목', 'Metric', '评分项目'),
    'total_header': ('누적', 'Total', '累计'),
    'gain_header': ('최근 1초 증가량', 'Last 1s gain', '最近1秒增量'),
    'rule_header': ('산정 기준 / 기록', 'Rule / record', '评分标准 / 记录'),
    'lap_pending': ('완주 시에만 기록 인정', 'Valid only after a full lap', '仅完成一圈后有效'),
    'progress': ('트랙 진행률', 'Track progress', '赛道进度'),
    'progress_rule': ('차선 이탈 없이 완주: 100점', 'Full lap within lane: 100', '不驶出车道完成一圈：100分'),
    'centering': ('중앙선 유지율', 'Lane centering', '中心线保持率'),
    'centering_rule': ('중앙 완주 100 · 경계 완주 20', 'Center lap 100 / edge lap 20', '中心完成100分 / 边缘完成20分'),
    'lap_time': ('랩타임', 'Lap time', '单圈时间'),
    'lap_record': ('인정 기록: {seconds:.2f}초', 'Valid lap: {seconds:.2f}s', '有效记录：{seconds:.2f}秒'),
    'recovering': ('차선 이탈 · {seconds:.1f}초 내 복귀하세요 · 이탈 구간 점수 제외', 'Off lane - return within {seconds:.1f}s - off-lane sections earn no points', '驶出车道 · 请在{seconds:.1f}秒内返回 · 驶出区间不计分'),
    'missing_font': ('GUI 글꼴을 찾을 수 없습니다. Noto Sans CJK를 설치하거나 language: eng로 설정하세요.', 'GUI font not found. Install Noto Sans CJK or set language: eng.', '找不到界面字体。请安装 Noto Sans CJK，或设置 language: eng。'),
}


def translate(key, language='kor', **values):
    return TEXT[key][LANGUAGES.index(language)].format(**values)
