# Changelog

## Unreleased

### Added
- Callback date: `request_callback` takes an optional `callback_at` the model works out from the date and time
  it is told ("завтра в 3" -> "22/10 15:00"); the runtime checks it (a real date from today up to 92 days ahead,
  not an hour already past) and adds it to the result's `callback` as `2026-10-22T15:00` (`2026-10-22` without
  an hour). `preferred_time` stays as the caller said it and is still read back. Capability `callback_at`.
- Optional bundle field `timezone` (IANA name): the PBX time zone of the date and time the agent is told;
  without it, or with a zone this runtime does not know, `VA_TIMEZONE` is used.
- Restarts keep the agents: `deploy.sh --restart` saves the old instance's bundles to the data volume
  (`voice-agent-runtime save-bundles`, `VA_HANDOFF_FILE`), and the new instance loads them and deletes the file
  before it reports ready. A call that comes within 30 s after a start, before its PBX pushed the bundle again,
  waits up to 10 s for it instead of going to the fallback number at once (a call right after a restart heard
  "сервис временно недоступен" and went to a fallback number nobody answered).
- Failover between the LLM providers for every call (`ProviderRouter`): after `LLM_FAILOVER_AFTER` (3) slow
  or failed turns of the provider asked first within `LLM_FAILOVER_WINDOW_S` (300 s) every call asks the other
  one first, with no hedge wait; the primary is probed every `LLM_RECOVERY_PROBE_S` (8 h) with
  `LLM_RECOVERY_PROBES` (3) short requests and gets the turns back when all answer in time. The topic filter
  follows the provider asked first. `/health` shows the state under `llm`.
- Topic filter (`topic.py`, `VA_TOPIC_FILTER`, on by default): a short side request to the primary provider
  (temperature 0, ~300 input / ~23 output tokens) per caller phrase decides whether it concerns the company. It
  runs in parallel with the answer (~0.2 s, before the answer's first token in almost every turn) and holds its
  audio back until the verdict; an off-topic phrase or an attempt to change the agent's rules gets a fixed
  refusal, the third in a row an offer of a person. Short replies and answers to "name / number / time / topic"
  questions are not checked; an error or a late verdict lets the answer through. Its tokens count in usage.
  The LLM connections of a call are opened while the greeting plays.
- The agent knows the date, the day of the week and the time (`VA_TIMEZONE`, default Europe/Moscow): call 17
  answered "да" to "сейчас рабочее время?" on a Sunday.
- Updates without dropping calls: `compose.bluegreen.yaml` (Caddy in front of `runtime-blue` / `runtime-green`)
  and `scripts/deploy.sh`. A new instance takes the bundles over from the running one (`VA_PEER_URL`,
  `GET /internal/handoff`, memory to memory) and warms the greetings before `GET /ready` answers 200; the old
  one is drained (`POST /internal/drain`, `voice-agent-runtime drain`) while results and acknowledgements of
  its calls are forwarded from the new one. `/health` gives the PBX `boot_id` and `capabilities`;
  `voice-agent-runtime status` and `healthcheck --ready`. Version 0.2.0.
- Silero TTS voices (v5_cis_base, MIT, stress by silero-stress): `silero-ekaterina`, `silero-oksana`,
  `silero-zinaida`, `silero-roman`, `silero-eduard`, 3-4 times faster than Piper. `VA_VOICES` selects the loaded
  voices; `GET /v1/voices` and `GET /v1/voices/{id}/sample` (WAV) for the module voice picker and preview.
  A bundle with a voice the runtime does not load speaks with `VA_DEFAULT_VOICE` instead of being rejected.
- Incremental speech recognition of long turns (`incremental.py`): each speculation recognizes only the audio
  after the last internal pause of at least `VA_STT_CHUNK_PAUSE_MS` (160); such pieces are committed while the
  caller keeps talking, so the end of the turn waits for the last phrase only and every second of audio is
  recognized about once. `VA_STT_CHUNKS`, `VA_STT_CHUNK_PAUSE_MS`, `VA_STT_CHUNK_MIN_MS`. New turn-detector
  event `CHUNK`; speculation, turn end and barge-in timing are unchanged.
- Hotwords: optional bundle field `hotwords` (list of strings, at most 200, each at most 100 characters) plus
  terms derived from the bundle bias GigaAM's greedy transducer decoder (`biasing.py`, the LocalSTTWorker
  approach, passed per call through onnx-asr kwargs so concurrent calls do not share state) and restore
  abbreviations that lost their first letter ("ТС" -> "АТС"). `VA_HOTWORDS`, `VA_HOTWORD_BOOST`.
- Usage metering (`usage.py`): per day, tenant and agent - calls, connected and agent speaking seconds, caller
  turns, LLM requests and input/output tokens (from `stream_options.include_usage`, estimated otherwise), TTS
  characters, fallbacks. Aggregates only, in SQLite `VA_USAGE_DB` (`/data` volume), flushed every
  `VA_USAGE_FLUSH_SECONDS`. `GET /v1/usage?from=&to=` for the PBX, `voice_agent_usage_*_total{tenant}` in
  `/metrics`. `LLM_STREAM_USAGE` / `LLM2_STREAM_USAGE` turn the usage request off.

### Changed
- Call 17: a farewell the model said after "Чем ещё могу помочь?" ("Всё так." -> "Спасибо, всего доброго!")
  ends the call instead of going on with "Хорошо. Чем ещё могу помочь?"; no second goodbye phrase after the
  model's own. A callback the caller asked for earlier in the call is not offered again; a placeholder time
  ("не уточнено") is asked for.
- Call 18: a callback topic that names something ("заявка на обратный звонок с отделом продаж") is kept; while
  a callback is being arranged a transfer the caller did not ask for becomes an offer; the caller interrupting
  a goodbye or transfer phrase ("Стой.") cancels the hangup; "какой сегодня день" passes the topic filter.
- Call 19: an unrequested transfer turned into an offer names the employee or department; a stutter of the
  model ("внутренний внутренний внутренний...") is not said; the callback time is asked with an open question
  and taken as said; complaints to the agent pass the topic filter.
- Call 20: "Да всё, наверное, больше ничем." and "Больше ничем." after "Чем ещё могу помочь?" end the call: the
  refusal may start with filler words ("больше", "наверное", "всё", "спасибо"...) and a lone "Да всё." counts too,
  instead of the agent asking "Хорошо. Чем ещё могу помочь?" again ("Всё так." still does not end it).
- Echo and backchannels over the talking agent (call of 2026-10-20 15:07: the echo "Здравствуйте" with the
  caller's "ну" cut the agent's answer off twice): the agent stops only for a word of the caller's own - its own
  recent words and backchannels ("ну", "угу", "да", "алло"...) are taken out word by word, stop words ("стоп",
  "подождите", "нет") always stop it - and such a phrase is not answered while the agent talks ("Да." right
  after its question still is). In the first 2 s of a reply (`BARGE_IN_HOLD_S`) an interruption waits for the
  recognized words, up to 3 s, instead of stopping the agent when recognition is slow.
- Wideband calls: `MEDIA_START format:slin16` (the module dials `c(slin16)` for G.722 / Opus callers) is
  accepted. The agent's audio is synthesized at 16 kHz (Silero at 24 kHz, Piper at its own rate, resampled), so
  those callers hear it without the 4 kHz telephone cut-off; the caller's audio is brought down to 8 kHz
  (`Downsampler`) before the echo canceller, VAD and recognition, which work as before. Frames are paced by their
  real duration. Capability `slin16` in `/health`; the phrase cache keeps each rate apart.
- A hand-set stress mark ("+" before a vowel, "МИК+О") in the agent texts now reaches Silero instead of being
  turned into a space; Piper and recognition hints drop it.
- The agent ends the call on an explicit request ("заверши звонок", "сбрось", "кладу трубку", "до встречи"), not
  only on a farewell. Narration between single asterisks ("*Звонок завершён.*") is not said; a reply left empty
  asks what else the caller needs.
- A turn whose LLM request failed before anything was said is asked once more instead of handing the caller to a
  person at once (a provider 500 on a malformed model message ended a call).
- A callback to a short internal caller number ("333") is accepted; before, the agent kept asking for the phone.
- Echo of the agent from a caller's speakerphone no longer interrupts it or gets an answer: an interruption is
  confirmed by recognition (at most 0.8 s, then it interrupts as before), and caller speech that started while
  the agent was audible and is only the agent's recent words is ignored (`echo.py`, metrics `echo_ignored`,
  `barge_ins_ignored`). Before, the greeting was cut after "Здравствуйте" and the agent answered itself twice.
- Echo cancellation: the caller audio goes through WebRTC AEC3 (livekit, Apache-2.0; `VA_AEC`) with the agent
  audio as the reference before VAD and recognition, and speech over the agent that it took 6 dB or more off is
  echo. On three recorded speakerphone calls it told all 7 echo pieces (7.9-34.5 dB off) from all 4 caller
  phrases (0.1-0.8 dB off).
- An interruption is checked once per phrase on its own audio (0.6 s of it); before, every voiced hop over the
  agent re-recognized the call from its start and blocked recognition of the caller's next phrases.
- A transfer on a question about the options ("А какие отделы есть?", "С кем можешь соединить?") does not
  happen: the agent names the departments from the bundle queues. No second offer when the reply already asks;
  tool arguments printed as text are not said; the prompt forbids "соединяю" without a transfer and invented
  modules, tariffs and conditions.
- The call ends on "хватит уже", "уйди", "отстань", "всё, спасибо", and on "ничем" / "нет" / "ничего не надо"
  answering "Чем ещё могу помочь?" (not answering an offer such as "Соединить вас?"); before, the agent kept
  asking "Чем ещё могу помочь?" for minutes.
- A callback request with a name the caller never said ("Клиент") or a time when the caller named none (a
  made-up "сейчас") is not saved: the agent asks for it. "Обратный звонок" is not a topic. No second
  "Записал вашу заявку" when the model already confirmed. Special tokens and filler (`<|start|>assistant`,
  `(ожидаем)......`) are not said.
- The prompt keeps the agent in its role: requests to forget or change the rules, switch roles, retell the
  instructions or talk about unrelated things get one polite refusal and a question about the company's
  business; the company's own extra rules still apply.
- `scripts/deploy.sh --restart` waits for a moment without calls and without uncollected results; before, a
  result the PBX came for just after the call ended was lost (the draining instance was no longer ready for the
  proxy and there was no second instance).
- Callback requests: nothing is saved while the agent's reply still asks something; the saved request is read
  back ("Записываю заявку: Егор, перезвоним завтра в 14:00 на номер …. Если что-то не так, поправьте")
  instead of a fixed phrase; name and time are asked in one question; a time counts only when the caller named
  a time (phone digits no longer do); another number needs 10 digits or more ("продиктуйте полностью");
  zero-width characters are not spoken. "Чем ещё могу помочь?" counts only as the reply's last question (a
  "нет" to "Удобно сейчас?" ended a call).
- A clause that is an English reasoning note ("Proceed with request_callback using phone 333.") is not said.
- `SpeechEngines.recognize(audio, terms=())` takes the agent's terms; `Turn` receives the futures of its
  recognized pieces instead of the audio.
- compose: `data` volume at `/data`; Dockerfile declares it.
