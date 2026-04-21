import smtplib
import ssl
from email.message import EmailMessage

from activetigger.config import config
from activetigger.datamodels import MessagesOutModel
from activetigger.db.manager import DatabaseManager


class Messages:
    """
    Manage messages on the interface
    - user messages
    - mail messages
    """

    mail_available: bool = config.mail_available
    mail_server: str | None = config.mail_server
    mail_server_port: int = config.mail_server_port
    mail_account: str | None = config.mail_account
    mail_password: str | None = config.mail_password

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        if self.mail_available:
            print("Mail service is available")
        else:
            print("Mail service is not available")

    def send_mail(self, to: str, subject: str, body: str):
        """
        Send a mail
        """
        if self.mail_server is None:
            raise Exception("Mail server is not configured")
        if self.mail_account is None:
            raise Exception("Mail account is not configured")
        if self.mail_password is None:
            raise Exception("Mail password is not configured")
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"Active Tigger <{self.mail_account}>"
        msg["To"] = to
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.mail_server, self.mail_server_port, context=context) as server:
            server.login(self.mail_account, self.mail_password)
            result = server.send_message(msg)

        if result != {}:
            print(f"Failed to send email to {to}: {result}")

    def send_mail_reset_password(self, user_name: str, mail: str, new_password: str) -> None:
        """
        Send a mail to reset the password
        """
        if not self.mail_available:
            raise Exception("Mail service is not available")

        subject = "Active Tigger - Password Reset"
        body = f"""
        Hello,

        Your password has been reset for the account : {user_name}
        
        Your new password is: {new_password}

        Please log in and change your password as soon as possible.

        Best regards,
        The Active Tigger Team
        """
        self.send_mail(mail, subject, body)

    def get_messages_system(self, from_user: str | None = None) -> list[MessagesOutModel]:
        """
        Get all system messages ordered by time desc
        """
        r = self.db_manager.messages_service.get_messages_system(from_user)
        return [
            MessagesOutModel(
                content=m.content, time=str(m.time), id=m.id, created_by=m.created_by, kind=m.kind
            )
            for m in r
        ]

    def get_messages_for_project(self, project_slug: str) -> list[MessagesOutModel]:
        """
        Get all project messages for a specific project ordered by time desc
        """
        r = self.db_manager.messages_service.get_messages_for_project(project_slug)
        return [
            MessagesOutModel(
                content=m.content, time=str(m.time), id=m.id, created_by=m.created_by, kind=m.kind
            )
            for m in r
        ]

    def get_messages_for_user(self, user_name: str) -> list[MessagesOutModel]:
        """
        Get all user messages for a specific user ordered by time desc
        """
        r = self.db_manager.messages_service.get_messages_for_user(user_name)
        return [
            MessagesOutModel(
                content=m.content, time=str(m.time), id=m.id, created_by=m.created_by, kind=m.kind
            )
            for m in r
        ]

    def get_messages(
        self,
        kind: str,
        from_user: str | None = None,
        for_user: str | None = None,
        for_project: str | None = None,
    ) -> list[MessagesOutModel]:
        """
        Get messages
        """
        if kind == "system":
            return self.get_messages_system()
        else:
            raise Exception(f"Unknown message kind: {kind}")

    def add_message(
        self, user_name: str, kind: str, content: str, property: dict | None = None
    ) -> None:
        """
        Add a message
        """
        self.db_manager.messages_service.add_message(
            user_name=user_name, kind=kind, property=property, content=content
        )

    def delete_message(self, id: int) -> None:
        """
        Delete a message by its ID.
        """
        self.db_manager.messages_service.delete_message(id)
