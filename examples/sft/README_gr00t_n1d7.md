# GR00T N1.7 supervised fine-tuning

The existing VLA SFT runner supports native GR00T N1.7 loss through
`actor.model.gr00t_sft`. This opts into a supervised model adapter; the existing
GR00T RL model and observation converters are unchanged.

Use `config/g1_sft_gr00t_n1d7.yaml` and provide a local model path, LeRobot v2
dataset, and a native GR00T processor directory with the desired modality
configuration, statistics and embodiment ID. The processor determines camera,
state/action slices and horizon. Preflight the dataset with NVIDIA's native
trainer before moving it to RLinf. Mixed-aspect views require equal-sized
preprocessing, e.g. native letterboxing saved in processor_config.json.

Use a GR00T N1.7-compatible environment (Python 3.12, Torch 2.9,
Transformers 4.57.3, TorchCodec 0.8, FFmpeg <8 and compatible FlashAttention)
with RLinf installed. The generic embodied installer's GR00T branch is an RL
simulation recipe, not an SFT-only installer. No simulator or live controller
is needed for offline SFT.

`batches_per_epoch` explicitly limits each loader iteration; the native stream
is cycled if necessary. GR00T partitions its own shards among ranks, so no extra
DistributedSampler is applied. Supply at least as many shards as actor ranks.
The initial smoke uses one rank and batch one. Distributed throughput/parity
has not been validated. Exact iterator resume and validation datasets are
rejected until they have a supported contract; checkpoints include the native
processor in addition to RLinf's model/optimizer state.

The example freezes the VLM and diffusion blocks, updating projection layers.
It is a wiring smoke configuration, not a recommended training recipe.
