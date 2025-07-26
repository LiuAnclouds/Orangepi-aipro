sudo systemctl daemon-reload
sudo systemctl enable mediamtx.service
sudo systemctl enable ffmpeg_stream.service
sudo systemctl start mediamtx.service
sudo systemctl start ffmpeg_stream.service