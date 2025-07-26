sudo systemctl daemon-reload
sudo systemctl enable mediamtx.service
sudo systemctl enable detect_om_camera.service
sudo systemctl start mediamtx.service
sudo systemctl start detect_om_camera.service
sudo systemctl status mediamtx.service
sudo systemctl status detect_om_camera.service