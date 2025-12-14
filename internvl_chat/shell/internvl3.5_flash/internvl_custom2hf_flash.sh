#!/bin/bash

# Define model sizes
MODEL_SIZES=(1 2 4)

# Loop through each model size
for SIZE in "${MODEL_SIZES[@]}"
do
    echo "Processing InternVL3.5-${SIZE}B model..."
    
    REMOTE_HF_PATH=OpenGVLab/InternVL3_5-${SIZE}B-HF # Use nonflash model config as init.
    LOCAL_HF_PATH=weights/InternVL3_5_Flash-${SIZE}B-HF
    REMOTE_CUSTOM_PATH=OpenGVLab/InternVL3_5-${SIZE}B-Flash
    LOCAL_CUSTOM_PATH=weights/InternVL3_5-${SIZE}B-Flash
    # Step 1: Download hf model config
    hf download ${REMOTE_HF_PATH} \
        config.json \
        preprocessor_config.json \
        processor_config.json \
        video_preprocessor_config.json \
        README.md \
        tokenizer.json \
        tokenizer_config.json \
        chat_template.jinja \
        added_tokens.json \
        special_tokens_map.json \
        vocab.json \
        --local-dir ${LOCAL_HF_PATH}

    # Step 2: Download custom model
    hf download ${REMOTE_CUSTOM_PATH} \
        --local-dir ${LOCAL_CUSTOM_PATH}
    
    # Step 3: Replace model config from nonflash to flash version.
    sed -i 's/InternVLForConditionalGeneration/InternVLFlashForConditionalGeneration/g' "${LOCAL_HF_PATH}/config.json"
    sed -i 's/"model_type": "internvl"/"model_type": "internvl_flash"/g' "${LOCAL_HF_PATH}/config.json"
    sed -i 's/InternVisionModel/InternvlFlashVisionModel/g' "${LOCAL_HF_PATH}/config.json"
    sed -i 's/"model_type": "internvl_vision"/"model_type": "internvl_flash_vision"/g' "${LOCAL_HF_PATH}/config.json"
    # Step 4: Add flash threshold fields from custom config to HF config
    FLASH_REL_THRESHOLD=$(jq -r '.flash_relative_threshold' "${LOCAL_CUSTOM_PATH}/config.json")
    FLASH_ABS_THRESHOLD=$(jq -r '.flash_absolute_threshold' "${LOCAL_CUSTOM_PATH}/config.json")
    
    # Add the fields to HF config using jq
    jq --arg rel "$FLASH_REL_THRESHOLD" --arg abs "$FLASH_ABS_THRESHOLD" \
        '. + {flash_relative_threshold: ($rel | tonumber), flash_absolute_threshold: ($abs | tonumber)}' \
        "${LOCAL_HF_PATH}/config.json" > "${LOCAL_HF_PATH}/config.json.tmp" && \
        mv "${LOCAL_HF_PATH}/config.json.tmp" "${LOCAL_HF_PATH}/config.json"
    
    echo "Added flash_relative_threshold: ${FLASH_REL_THRESHOLD}"
    echo "Added flash_absolute_threshold: ${FLASH_ABS_THRESHOLD}"


    # Step 4: transform model
    python internvl_chat/tools/internvl_custom2hf_flash.py \
        --custom_path ${LOCAL_CUSTOM_PATH} \
        --hf_path ${LOCAL_HF_PATH} \
        --save_path ${LOCAL_HF_PATH}
    
done

echo "All models processed successfully!"