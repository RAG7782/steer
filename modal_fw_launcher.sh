#!/bin/bash
# Launch all FW tests as separate Modal functions (each survives independently)
cd ~/a2rag

# FW6: 3 whitening jobs
modal run modal_future_work_tests.py::fw6_isotropy_whitening --model-name "BAAI/bge-small-en-v1.5" &
modal run modal_future_work_tests.py::fw6_isotropy_whitening --model-name "intfloat/e5-small-v2" &
modal run modal_future_work_tests.py::fw6_isotropy_whitening --model-name "thenlper/gte-small" &

# FW10: Large LLM preprocessing (L4 GPU)
modal run modal_future_work_tests.py::fw10_preprocessing_large_llm &

# FW12: Conceptors comprehensive
modal run modal_future_work_tests.py::fw12_conceptors_comprehensive &

# FW15: 6 addition vs nlerp jobs
for m in "all-MiniLM-L6-v2" "BAAI/bge-small-en-v1.5" "all-mpnet-base-v2" "BAAI/bge-base-en-v1.5" "intfloat/e5-small-v2" "thenlper/gte-small"; do
    modal run modal_future_work_tests.py::fw15_addition_vs_nlerp --model-name "$m" &
done

echo "All FW tests launched ($(jobs -r | wc -l) background jobs)"
wait
echo "All FW tests completed"
