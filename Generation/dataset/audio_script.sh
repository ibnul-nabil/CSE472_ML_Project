mkdir -p mozilla_s1_fixed
for f in mozilla_s1/*.wav; do
    ffmpeg -i "$f" -ar 22050 -ac 1 -sample_fmt s16 "mozilla_s1_fixed/$(basename $f)" -y -loglevel error
done

# fix ljspeech metadata format
#awk -F'|' '{print $1"|"$2"|"$2}' metadata.csv > metadata_fixed.csv


#Generation/dataset/mozilla_s1_fixed/wavs/common_voice_s1_907.wav
#Generation/dataset/mozilla_s1_fixed/wavs/common_voice_s1_288.wav