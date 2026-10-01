from demo import run_demo


def test_offline_demo(tmp_path):
    output = tmp_path / 'demo.avi'
    result = run_demo(output)
    assert result == {'synthetic': True, 'frames': 32, 'alerts': 1,
                      'final_track_ids': [1], 'local_alert_events': 1, 'alert_delivery': 'local_callback'}
    assert output.stat().st_size > 0
