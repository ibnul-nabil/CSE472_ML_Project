import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits,VitsArgs,VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# ============================================================
# 🔧 EDIT THESE PATHS
# ============================================================
BASE_DIR       = "/home/tr/MEGA/programming2/L4T2/CSE472_ML_Project"
RESTORE_PATH   = f"{BASE_DIR}/Generation/best_model.pth"
DATA_PATH      = f"{BASE_DIR}/Generation/dataset/mozilla_s1/"
META_TRAIN     = f"{BASE_DIR}/Generation/dataset/mozilla_s1/metadata_fixed.csv"
OUTPUT_PATH    = f"{BASE_DIR}/Generation/tuned_model"

# ============================================================
# ⚙️ CONFIG
# ============================================================
config = VitsConfig(
    # --- Run ---
    run_name="male_vits_finetune_v1",
    output_path=OUTPUT_PATH,

    # --- Training ---
    epochs=200,
    batch_size=16,
    eval_batch_size=16,
    grad_clip=[5.0, 5.0],
    lr_gen=2e-5,
    lr_disc=2e-5,
    lr_scheduler_gen="ExponentialLR",
    lr_scheduler_gen_params={"gamma": 0.9999, "last_epoch": -1},
    lr_scheduler_disc="ExponentialLR",
    lr_scheduler_disc_params={"gamma": 0.9999, "last_epoch": -1},

    # --- Logging ---
    print_step=100,
    plot_step=50,
    save_step=1000,
    save_n_checkpoints=5,
    save_checkpoints=True,
    save_all_best=True,
    save_best_after=500,
    print_eval=True,
    run_eval=True,
    dashboard_logger="tensorboard",

    # --- Audio ---
    audio=VitsAudioConfig(
        fft_size=1024,
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    ),

    # --- Text / Phonemes ---
    use_phonemes=True,
    phonemizer="espeak",
    phoneme_language="bn",
    compute_input_seq_cache=True,
    text_cleaner="phoneme_cleaners",
    add_blank=True,
    phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),

    # --- Loss weights ---
    kl_loss_alpha=1.0,
    disc_loss_alpha=1.0,
    gen_loss_alpha=1.0,
    feat_loss_alpha=1.0,
    mel_loss_alpha=80.0,
    dur_loss_alpha=1.0,

    # --- Data ---
    min_audio_len=1,
    max_audio_len=500000,
    min_text_len=1,
    compute_linear_spec=True,
    return_wav=True,
    shuffle=True,
    drop_last=True,
    eval_split_size=0.05,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    use_language_weighted_sampler=True,

    # --- Dataset ---
    datasets=[
        dict(
            formatter="ljspeech",
            dataset_name="mozilla_s1",
            path=DATA_PATH,
            meta_file_train=META_TRAIN,
            meta_file_val="",
            ignored_speakers=None,
            language="",
            phonemizer="",
            meta_file_attn_mask="",
        )
    ],

    # --- Test sentences ---
    test_sentences=[
        "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।",
        "চিরদিন তোমার আকাশ, তোমার বাতাস, আমার প্রাণে বাজায় বাঁশি",
    ],

    # --- Model args ---
    model_args=VitsArgs(
        num_chars=131,
        out_channels=513,
        spec_segment_size=32,
        hidden_channels=192,
        hidden_channels_ffn_text_encoder=768,
        num_heads_text_encoder=2,
        num_layers_text_encoder=6,
        kernel_size_text_encoder=3,
        dropout_p_text_encoder=0.1,
        dropout_p_duration_predictor=0.5,
        kernel_size_posterior_encoder=5,
        dilation_rate_posterior_encoder=1,
        num_layers_posterior_encoder=16,
        kernel_size_flow=5,
        dilation_rate_flow=1,
        num_layers_flow=4,
        resblock_type_decoder="1",
        resblock_kernel_sizes_decoder=[3, 7, 11],
        resblock_dilation_sizes_decoder=[[1,3,5],[1,3,5],[1,3,5]],
        upsample_rates_decoder=[8, 8, 2, 2],
        upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[16, 16, 4, 4],
        periods_multi_period_discriminator=[2, 3, 5, 7, 11],
        use_sdp=True,
        noise_scale=1.0,
        inference_noise_scale=0.5,
        length_scale=1.0,
        noise_scale_dp=1.0,
        inference_noise_scale_dp=1.0,
        init_discriminator=True,
        use_spectral_norm_disriminator=False,
        use_speaker_embedding=False,
        num_speakers=0,
        speakers_file=None,
        speaker_embedding_channels=256,
        use_d_vector_file=False,
        d_vector_dim=0,
        detach_dp_input=True,
        use_language_embedding=False,
        embedded_language_dim=4,
        num_languages=0,
        condition_dp_on_speaker=True,
        freeze_encoder=False,
        freeze_DP=False,
        freeze_PE=True,
        freeze_flow_decoder=False,
        freeze_waveform_decoder=False,
        interpolate_z=True,
        reinit_DP=False,
        reinit_text_encoder=False,
    ),
)

# ============================================================
# 🚀 INIT & TRAIN
# ============================================================
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(config.phoneme_cache_path, exist_ok=True)

# Save config for reference
config.save_json(os.path.join(OUTPUT_PATH, "finetune_config.json"))

# Build model components
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)
model = Vits(config, ap, tokenizer, speaker_manager=None)

# Trainer args — restore_path loads the checkpoint
trainer_args = TrainerArgs(
    restore_path=RESTORE_PATH,
    skip_train_epoch=False,
)

trainer = Trainer(
    trainer_args,
    config,
    output_path=OUTPUT_PATH,
    model=model,
)

trainer.fit()