import unittest
from parameterized import parameterized

from bridger.logging import object_logger

_TMP_DIR = "tmp"

# TODO(lyric): Refactor to test_util.
def clean_up_dir(path):
    for filepath in path.iterdir():
        filepath.unlink()

def create_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    clean_up_dir(path)


def delete_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    clean_up_dir(path)
    path.rmdir()

class TestObjectLogger(unittest.TestCase):

    def setUp(self):
        create_temp_dir()

    def tearDown(self):
        delete_temp_dir()
    
    @parameterized.expand([
        (1,),(2,)]
    )
    def test_object_logger(self, buffer_size):
        test_filepath = os.path.join(_TMP_DIR, "test")
        with object_logger(dirname=_TMP_DIR, log_filename="test",buffer_size=buffer_size):
            log("a")
            print(os.path.getsize(test_filepath))
            

if __name__ == "__main__":
    unittest.main()



