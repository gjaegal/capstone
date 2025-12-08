
import subprocess
import time
import os

# --- ⚙️ 사용자 설정 (필수!) ---
# 1. 사용 중인 ROS 버전에 맞게 수정하세요. (noetic, melodic 등)
ROS_DISTRO = "noetic"

# 2. 메인으로 사용하는 catkin 워크스페이스의 절대 경로를 입력하세요.
# 예: "/root/catkin_ws", "/home/your_user/catkin_ws"
# 경로를 모를 경우, Ubuntu 터미널에서 'ls -d ~/*_ws/' 명령어로 찾아보세요.
# ROS_WORKSPACE_PATH = "/root/catkin_ws" 
ROS_WORKSPACE_PATH = "/home/password1234/catkin_ws"  # 절대경로 설정 필수!


# 3. 터미널 실행 간 지연 시간 (초)
DELAY_BETWEEN_TERMINALS = 3.0

# --- 스크립트 설정 ---
# ROS 환경을 로드하는 명령어 문자열을 생성합니다.
# 워크스페이스의 setup.bash를 로드하면 ROS 기본 설정도 함께 로드됩니다.
ros_setup_command = f"source {ROS_WORKSPACE_PATH}/devel/setup.bash"

# 실행할 명령어 목록
commands = [
    # 1. 첫번째 터미널: roscore 실행
    f'{ros_setup_command}; roscore; exec bash',

    # 2. 두번째 터미널: Kobuki 노드 실행
    f'{ros_setup_command}; usbipd.exe attach --wsl --busid 1-2 || true; roslaunch kobuki_node minimal.launch; exec bash',

    # 3. 세번째 터미널: 카메라 스트림 실행
    # 'cd' 명령어를 포함하므로, setup 이후에 실행되도록 구성합니다.
    f'{ros_setup_command}; usbipd.exe attach --wsl --busid 1-13 || true; cd project; python3 main.py; exec bash',

    # 4. 네번째 터미널: 컨트롤러 실행
    f'{ros_setup_command}; cd {ROS_WORKSPACE_PATH}/src/yolo_kobuki_controller/scripts; python3 controller4.py; exec bash'
]

# --- 스크립트 실행 본문 ---
def main():
    print("ROS 자동 실행 스크립트를 시작합니다...")
    
    # 설정된 경로가 존재하는지 간단히 확인
    print(f"설정된 워크스페이스 경로: {ROS_WORKSPACE_PATH}")
    print(f"ROS 환경 설정 명령어: '{ros_setup_command}'")
    
    for i, cmd in enumerate(commands):
        terminal_num = i + 1
        print(f"\n[{terminal_num}/{len(commands)}] 번째 터미널을 실행합니다.")
        
        wsl_command = f'start wsl.exe -d Ubuntu-20.04 -u password1234 --cd ~ -e bash -c "{cmd}"'
        
        try:
            subprocess.Popen(wsl_command, shell=True)
        except Exception as e:
            print(f"오류: {terminal_num}번째 터미널을 시작하지 못했습니다: {e}")
            os.system("pause")
            return

        if terminal_num < len(commands):
            print(f"{DELAY_BETWEEN_TERMINALS}초 후 다음 터미널을 엽니다...")
            time.sleep(DELAY_BETWEEN_TERMINALS)
    
    print("\n모든 터미널 실행이 완료되었습니다.")

if __name__ == "__main__":
    main()
