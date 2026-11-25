# Telegram notification callback — `TelegramNotifierCallback`

`TelegramNotifierCallback`, registry key `telegram_notifier_callback`, creates a Telegram session at `on_fit_start`, tracks the best monitor in `on_epoch_end`, and sends one final message at `on_fit_end`.

| Parameter | Default |
| --- | --- |
| `token_env_field` | `TELEGRAM_BOT_TOKEN` |
| `chat_id_env_field` | `TELEGRAM_CHAT_ID` |
| `message_prefix` | `✅ Training finished` |
| `message_suffix` | `""` |
| `include_mlflow_experiment_name` | `True` |
| `include_trainer_progress` | `True` |
| `include_last_logs` / `include_best_logs` | `True` |
| `monitor` / `mode` / `min_delta` | `val/loss` / `min` / `0.0` |
| `request_timeout_sec` | `15` |
| `telegram_parse_mode` | `HTML` |

The callback requires the named environment variables and the `requests` dependency. It closes the session after sending; HTTP failures warn rather than raising from the send path. Never put tokens in YAML or source control.

```bash
export TELEGRAM_BOT_TOKEN='your-token'
export TELEGRAM_CHAT_ID='your-chat-id'
```

```yaml
- name: telegram_notifier_callback
  params: {monitor: val/mae, mode: min, request_timeout_sec: 15}
```
