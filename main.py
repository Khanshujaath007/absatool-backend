from pyabsa import (
    available_checkpoints,
    ATEPCCheckpointManager,
)

"""
MLModel takes list of strings as parameter and performs the proceesing on data, and
return a object with aspect terms, sentiment of these aspects ans confidence of these terms 
processed by the model
"""


def aggregatePayload(result):
    payload = {
        "Aspect Terms": [],
        "Sentiment of Aspects": [],
        "Confidence of Aspect": [],
    }
    for i in range(len(result)):
        for j in range(len(result[i]["aspect"])):
            payload["Aspect Terms"].append(result[i]["aspect"][j])
            payload["Sentiment of Aspects"].append(result[i]["sentiment"][j])
            payload["Confidence of Aspect"].append(result[i]["confidence"][j])

    return payload


def MLModel(data):
    check_point_map = available_checkpoints()
    if data != []:
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint="english", auto_device=True
        )
        atepc_result = aspect_extractor.batch_predict(
            target_file=data, pred_sentiment=True, print_result=False, save_result=False
        )
        # iterate through all the atepc result objects and build payload
        finalPayload = aggregatePayload(atepc_result)
        return finalPayload
    else:
        return {"response": "Data not found"}


__all__ = ["MLModel"]
