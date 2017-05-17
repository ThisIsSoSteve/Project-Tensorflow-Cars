Installing vXbox
goto http://vjoystick.sourceforge.net/site/index.php/vxbox

download the correct scpvbus 64/32
Bus Installation
IMPORTANT: (Windows 7 Only) You must install the latest Xbox 360 drivers first.

Unzip bus package into a temporary folder (scpvbus-x64.zip for a 64-bit target machine, scpvbus-x86.zip for a 32-bit target machine ).
Open a command window (CMD) as an administrator.
CD to the above temporary folder.
Enter the following command: devcon.exe install ScpVBus.inf Root\ScpVBus



Bus Removal
Same as installation, only a different command: devcon.exe remove Root\ScpVBus
Note: You cannot remove a bus if a vXbox device is plugged in to it.

download vXboxInterface.dll 64/32

https://github.com/shauleiz/vXboxInterface/releases

python

pip install PYXInput
pip uninstall PYXInput


