import subprocess
from datetime import datetime

import pandas as pd
from feast import FeatureStore
from feast.data_source import PushMode


def run_workflow():
    store = FeatureStore(repo_path=".")
    print("\n--- Run feast apply ---")
    subprocess.run(["feast", "apply"])

    print("\n--- Historical features for training ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=False)

    print("\n--- Historical features for batch scoring ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=True)

    print("\n--- Load features into online store ---")
    store.materialize_incremental(end_date=datetime.now())

    print("\n--- Simulate a stream event ingestion of the hourly stats df ---")
    event_df = pd.DataFrame.from_dict(
        {
            "url": ["https://t.me/s/jobs_in_it_remoute/24798"],
            "event_timestamp": [
                datetime.now(),
            ],
            "content": ["some information"],
            "outlinks": ["none"],
            "linkPreview": ["none"],
        }
    )
    print(event_df)
    store.push("posts_push_source", event_df, to=PushMode.ONLINE_AND_OFFLINE)

    print("\n--- Online features again with updated values from a stream push---")
    fetch_online_features(store, source="push")

    print("\n--- Run feast teardown ---")
    subprocess.run(["feast", "teardown"])


def fetch_historical_features_entity_df(store: FeatureStore, for_batch_scoring: bool):
    # Note: see https://docs.feast.dev/getting-started/concepts/feature-retrieval for more details on how to retrieve
    # for all entities in the offline store instead
    entity_df = pd.DataFrame.from_dict(
        {
            # entity's join key -> entity values
            "driver_id": [1001, 1002, 1003],
            # "event_timestamp" (reserved key) -> timestamps
            "event_timestamp": [
                datetime(2024, 4, 12, 10, 59, 42),
                datetime(2024, 4, 12, 8, 12, 10),
                datetime(202, 4, 12, 16, 40, 26),
            ],
            "val_to_add": [1, 2, 3],
        }
    )
    # For batch scoring, we want the latest timestamps
    if for_batch_scoring:
        entity_df["date"] = pd.to_datetime("now", utc=True)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "posts_hourly_stats:content",
            "posts_hourly_stats:outlinks",
            "posts_hourly_stats:linksPreview",
            "transform_content:preprocess_content",
        ],
    ).to_df()
    print(training_df.head())


if __name__ == "__main__":
    run_workflow()
