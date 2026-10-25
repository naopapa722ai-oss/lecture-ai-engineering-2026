"""質問スクリプトのテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

import app.ask as ask_module


def test_chat_keeps_history_between_questions(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.messages = []

        def send_message(self, message):
            self.messages.append(message)
            return SimpleNamespace(
                text=f"回答: {message}",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1,
                    candidates_token_count=2,
                ),
            )

    class FakeChats:
        def __init__(self, conversation):
            self.conversation = conversation
            self.create_args = None

        def create(self, **kwargs):
            self.create_args = kwargs
            return self.conversation

    conversation = FakeChat()
    chats = FakeChats(conversation)
    client = SimpleNamespace(chats=chats)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    questions = iter(["最初の質問", "続きの質問", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))

    ask_module.chat()

    assert conversation.messages == ["最初の質問", "続きの質問"]
    assert chats.create_args["model"] == ask_module.MODEL
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: 続きの質問" in output
