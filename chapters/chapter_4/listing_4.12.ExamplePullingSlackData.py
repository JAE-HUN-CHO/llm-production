from dotenv import load_dotenv
import os

import pandas
import slack_sdk

# 프로젝트 루트의 .env 를 환경변수로 로드
load_dotenv()

# 토큰은 환경변수 SLACK_TOKEN 으로 전달한다. 없으면 실제 Slack API 대신
# 저장소의 로컬 데이터(data/Slack_Dataset.csv)를 사용하는 mock 경로로 동작한다.
token_slack = os.environ.get("SLACK_TOKEN")

if token_slack:
    client = slack_sdk.WebClient(token=token_slack)

    auth = client.auth_test()
    self_user = auth["user_id"]

    dm_channels_response = client.conversations_list(types="im")

    all_messages = {}

    for channel in dm_channels_response["channels"]:
        history_response = client.conversations_history(
            channel=channel["id"]
        )
        all_messages[channel["id"]] = history_response["messages"]

    txts = []

    for channel_id, messages in all_messages.items():
        for message in messages:
            try:
                text = message["text"]
                user = message["user"]
                timestamp = message["ts"]
                txts.append([timestamp, user, text])
            except Exception:
                pass

    slack_dataset = pandas.DataFrame(txts)
    slack_dataset.columns = ["timestamp", "user", "text"]
    df = slack_dataset[slack_dataset.user == self_user]
else:
    print("SLACK_TOKEN not set: using local data/Slack_Dataset.csv")
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "data",
        "Slack_Dataset.csv",
    )
    df = pandas.read_csv(csv_path)

df[["text"]].to_parquet("slack_dataset.gzip", compression="gzip")
print(f"Saved {len(df)} messages to slack_dataset.gzip")
