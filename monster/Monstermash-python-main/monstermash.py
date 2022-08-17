from ctypes import *
import ctypes

# 키보드 키값 정의
SDLK_UNKNOWN = 0
SDLK_RETURN = '\r'
SDLK_ESCAPE = '\033'
SDLK_BACKSPACE = '\b'
SDLK_TAB = '\t'
SDLK_SPACE = ' '
SDLK_EXCLAIM = '!'
SDLK_QUOTEDBL = '"'
SDLK_HASH = '#'
SDLK_PERCENT = '%'
SDLK_DOLLAR = '$'
SDLK_AMPERSAND = '&'
SDLK_QUOTE = '\''
SDLK_LEFTPAREN = '('
SDLK_RIGHTPAREN = ')'
SDLK_ASTERISK = '*'
SDLK_PLUS = '+'
SDLK_COMMA = ''
SDLK_MINUS = '-'
SDLK_PERIOD = '.'
SDLK_SLASH = '/'
SDLK_0 = '0'
SDLK_1 = '1'
SDLK_2 = '2'
SDLK_3 = '3'
SDLK_4 = '4'
SDLK_5 = '5'
SDLK_6 = '6'
SDLK_7 = '7'
SDLK_8 = '8'
SDLK_9 = '9'
SDLK_COLON = ':'
SDLK_SEMICOLON = ';'
SDLK_LESS = '<'
SDLK_EQUALS = '='
SDLK_GREATER = '>'
SDLK_QUESTION = '?'
SDLK_AT = '@'
SDLK_LEFTBRACKET = '['
SDLK_BACKSLASH = '\\'
SDLK_RIGHTBRACKET = ']'
SDLK_CARET = '^'
SDLK_UNDERSCORE = '_'
SDLK_BACKQUOTE = '`'
SDLK_a = 'a'
SDLK_b = 'b'
SDLK_c = 'c'
SDLK_d = 'd'
SDLK_e = 'e'
SDLK_f = 'f'
SDLK_g = 'g'
SDLK_h = 'h'
SDLK_i = 'i'
SDLK_j = 'j'
SDLK_k = 'k'
SDLK_l = 'l'
SDLK_m = 'm'
SDLK_n = 'n'
SDLK_o = 'o'
SDLK_p = 'p'
SDLK_q = 'q'
SDLK_r = 'r'
SDLK_s = 's'
SDLK_t = 't'
SDLK_u = 'u'
SDLK_v = 'v'
SDLK_w = 'w'
SDLK_x = 'x'
SDLK_y = 'y'
SDLK_z = 'z'

# Monstermash 모드 정의
DRAW_OUTLINE = 0
DRAW_REGION_OUTLINE = 1
DRAW_REGION = 2
OUTLINE_RECOLOR = 3
REGION_REDRAW_MODE = 4
REGION_SWAP_MODE = 5
REGION_MOVE_MODE = 6
DEFORM_MODE = 7
ANIMATE_MODE = 8

# Callback 함수 포인터 정의
KEY_CALLBACK = CFUNCTYPE(None,ctypes.c_int,ctypes.py_object)

# C++에 전달할 문자열 변환
def convert_ctype(x):
    return x.encode('utf-8')

# C++에서 전달받은 문자열 Python에서 인식할 수 있도록 변환
def decode_ctype(x):
    return str(ctypes.c_char_p(x).value.decode('utf-8'))
 
class Monstermash(object):
    # __init__(self, w, h, title, shaders) : 객체생성
    # - w: 윈도우 폭
    # - h: 윈도우 높이
    # - title: 윈도우 명
    # - shader: inflate시 적용될 matrial 파일 -> 미적용 시 검은색으로 출력됨
    def __init__(self, w, h, title, shaders):
        self.lib = cdll.LoadLibrary('./monstermash.so') # so 파일 연결(monstermash.so 파일이 존재하는 경로 입력)
        mainWindow_new = self.lib.mainWindow_new
        mainWindow_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        mainWindow_new.restype = ctypes.c_void_p
        self.obj = mainWindow_new(w, h, title, shaders)
        self.setCallBack(self.keyResponse) # 키보드 입력 시 실행될 콜백 함수로 keyResponse 지정
    
    # run(self) : monstermash event loop 시작 -> 마우스, 키보드 입력에 반응시작
    def run(self):
        self.lib.mainWindow_runLoop(self.obj)

    # setCallBack(self, callback) : 키보드 입력 시 실행될 콜백함 수 지정
    # - callback : 콜백 함수 이름 -> 콜백함수는 void callback(int, py_object) 형태여야 함
    def setCallBack(self, callback):
        mainWindow_setCallBack = self.lib.mainWindow_setCallBack
        mainWindow_setCallBack(self.obj, callback, ctypes.py_object(self))

    # keyResponse(key, self) : 키보드 입력 시 실행될 콜백함수
    # - key : 키보드에서 입력된 키
    # - self : 콜백함수에서 사용될 객체 -> Monstermash 객체를 의미함
    @KEY_CALLBACK # keyResponse의 자료형을 callback 함수 형태로 정의
    def keyResponse(key, self):
        print(key) # 입력된 키보드 키 출력
        if chr(key) == SDLK_1: # 1 키 입력 시 
            self.change_draw_mode() # draw 모드로 전환
        elif chr(key) == SDLK_2: # 2 키 입력 시 
            self.change_inflate_mode() # inflate 모드로 전환
        elif chr(key) == SDLK_3: # 3 키 입력 시 
            self.save_project("./result.zip") # result.zip 파일로 프로젝트 저장
        elif chr(key) == SDLK_4: # 4 키 입력 시 
            self.open_project("./result.zip") # result.zip 파일로 저장된 프로젝트 열기
        elif chr(key) == SDLK_5: # 5 키 입력 시 
            self.load_template("./template.png") # template.png 파일을 template 이미지로 열기
        elif chr(key) == SDLK_6: # 6 키 입력 시 
            self.load_background("./template.png") # template.png 파일을 background 이미지로 열기
        elif chr(key) == SDLK_0: # 0 키 입력 시 
            self.reset() # 프로젝트 리셋

    # change_draw_mode(self) : draw 모드로 전환
    def change_draw_mode(self):
        self.lib.mainWindow_changeManipulationMode(self.obj, DRAW_OUTLINE)

    # change_inflate_mode(self) : inflate 모드로 전환
    def change_inflate_mode(self):
        self.lib.mainWindow_changeManipulationMode(self.obj, DEFORM_MODE)

    # save_project(self, project_name) : 프로젝트 zip 파일로 저장
    # - project_name : 프로젝트 저장경로 & 프로젝트 저장 명 -> ex) "./result.zip"
    def save_project(self, project_name):
        self.lib.mainWindow_saveProject(self.obj, convert_ctype(project_name))

    # open_project(self, project_name) : zip 파일로 저장된 프로젝트 열기
    # - project_name : 프로젝트 경로 & 프로젝트 저장 명 -> ex) "./result.zip"
    def open_project(self, project_name):
        self.lib.mainWindow_openProject(self.obj, convert_ctype(project_name))

    # reset(self) : 프로젝트 리셋
    def reset(self):
        self.lib.mainWindow_reset(self.obj)

    # load_template(self, img_path) : png 파일을 template 이미지로 열기
    # - img_path : 이미지 경로 & 이미지 저장 명 -> ex) "./template.png"
    def load_template(self, img_path):
        self.lib.mainWindow_loadTemplateImageFromFile(self.obj, convert_ctype(img_path))

    # load_background(self, img_path) : png 파일을 background 이미지로 열기
    # - img_path : 이미지 경로 & 이미지 저장 명 -> ex) "./template.png"
    def load_background(self, img_path):
        self.lib.mainWindow_loadBackgroundImageFromFile(self.obj, convert_ctype(img_path))
 
if __name__ == '__main__':
    # monstermash 객체생성 -> w:1000pixel, h:800pixel, title:monster mash, shaders:./matcapOrange.jpg
#    monstermash = Monstermash(1000, 800, convert_ctype("monster mash"), convert_ctype("./matcapOrange.jpg"))
    monstermash = Monstermash(717, 495, convert_ctype("monster mash"), convert_ctype("./matcapOrange.jpg"))
   
    # monstermash event loop 실행 -> 키보드, 마우스 이벤트에 반응
    monstermash.run()
