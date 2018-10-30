import os

from com.designingnn.client.core import AppContext


class StatusUpdateService:
    def __init__(self):
        self.status_file = os.path.join(AppContext.METADATA_DIR, 'status.txt')

        if not os.path.exists(self.status_file):
            print("status file not found creating one.")
            with open(self.status_file, 'w+') as status_f:
                status_f.write('{}'.format('free'))

    def get_client_status(self):
        status = None
        with open(self.status_file, 'r') as status_f:
            status = status_f.read()

        return status.strip()

    def update_client_status(self, status):
        with open(self.status_file, 'w+') as status_f:
            status_f.write('{}'.format(status))

if __name__ == '__main__':
    AppContext.METADATA_DIR = '/home/sai/mtss_proj/metadata'

    print StatusUpdateService().get_client_status()
    StatusUpdateService().update_client_status('busy')
    print StatusUpdateService().get_client_status()
