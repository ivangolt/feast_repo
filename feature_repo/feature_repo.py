from datetime import timedelta

import pandas as pd
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.data_format import ParquetFormat
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, String
from posts_preprocessing import text_preprocessing

# Define an entity for the post
post = Entity(name="posts", join_keys=["url"])

# Define a source for the feature data
feature_source = FileSource(
    name="posts_parser",
    path="data/new_posts.parquet",
    event_timestamp_column="date",
    file_format=ParquetFormat(),
)


# Define a Feature View
posts_features = FeatureView(
    name="posts_features",
    entities=["url"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="content", dtype=String),
        Field(name="outlinks", dtype=String),
        Field(name="linkPreview", dtype=String),
    ],
    # online=True,
    source=feature_source,
)


# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
input_request = RequestSource(
    name="vals_to_add",
    schema=[
        Field(name="val_to_add", dtype=Int64),
    ],
)


# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
@on_demand_feature_view(
    sources=[posts_features, input_request],
    schema=[
        Field(name="preprocess_content", dtype=Float64),
    ],
)
def transformed_content(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df["content"] = input_df["content"].apply(text_preprocessing)

    return input_df


# This groups features into a model version
posts_v1 = FeatureService(
    name="posts_v1",
    features=[
        posts_features[["content"]],  # Sub-selects a feature from a feature view
        transformed_content,  # Selects all features from the feature view
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="C:\dev\feast_repo\feature_repo\data")
    ),
)
posts_v2 = FeatureService(
    name="posts_v2", features=[posts_features, transformed_content]
)

# Defines a way to push data (to be available offline, online or both) into Feast.
posts_push_source = PushSource(name="posts_push_source", batch_source=feature_source)
