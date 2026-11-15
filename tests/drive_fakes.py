"""In-memory Drive client for tests that publish and restore company data."""

from tests.test_drive import FakeDriveClient


class MemoryDrive(FakeDriveClient):
    def __init__(self):
        super().__init__()
        self.content = {}
        self.corrupt = False
        self.fail_upload = None

    def create_file(self, parent_id, local_path, name):
        if name == self.fail_upload:
            raise RuntimeError("Simulated interruption")
        super().create_file(parent_id, local_path, name)
        item = self.find_child(parent_id, name)
        self.content[item.id] = local_path.read_bytes()

    def download_file(self, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupt" if self.corrupt else self.content[file_id])

    def update_file(self, *args):
        raise AssertionError("An immutable SEC source should never be updated")
