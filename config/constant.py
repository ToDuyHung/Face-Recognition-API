# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 1000

# Minimum length of time where no motion is detected it should take
#(in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 100

EMBEDDING_DIMENSION = 512

ID = "_id"
NAME = "name"
FEATURES = "features"

FILE_SERVER_URL= "http://172.28.0.23:35432/api/file/upload-file-local"
FILE_SERVER_GET_URL= "http://172.28.0.23:35432/api/file/Get-File-Local?guid="

URL_LOGIN_HRM = "https://api.hrm.rke.dev.tmtco.org/api/v1/sign-in/password"

HEADER_LOGIN = {
'Content-Type': 'application/json',
'Cookie': 'INGRESSCOOKIE=1659603241.501.23902.710292|ca59cbf6c990866117ea18a015436a0a'
}

URL_GET_ALL = "https://hrm.tmtco.org/hrm/api/v1/sync-data-AI/all-user"
COOKIE_GET_ALL = 'INGRESSCOOKIE=1659603241.501.23902.710292|ca59cbf6c990866117ea18a015436a0a'
API_KEY = 'YWhzYnZhc3Vhc2Jhc2Fpc2FzYXNhc2E='


KELVIN_TABLE = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}


FINAL_RMEAN = 172.305
FINAL_RSTD = 34.178
FINAL_GMEAN = 136.218
FINAL_GSTD = 31.839
FINAL_BMEAN = 128.57875
FINAL_BSTD = 32