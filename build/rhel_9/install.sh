sudo dnf update -y
sudo dnf install epel-release -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install python3 python3-devel git wget -y


python3 -m venv gpt-env
source gpt-env/bin/activate


pip install --upgrade pip
pip install torch torchvision torchaudio transformers



pip install diffusers
pip install torch pillow
pip install reportlab




#optiuonal for CUDA GPUS
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

#CPU
pip install torch torchvision torchaudio

pip install transformers


pip install bitsandbytes

pip install fastapi uvicorn


sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
