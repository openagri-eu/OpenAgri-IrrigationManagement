from typing import Generator
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from core import decode_token
from models import User
from crud import user

from core.config import settings
from db.session import SessionLocal

reusable_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/login/access-token/")


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def get_jwt(
        token: str = Depends(reusable_oauth2)
):
    if not token:
        raise HTTPException(
            status_code=403,
            detail="Not authenticated"
        )

    return token


# Only use when you're expecting an AT that came from IRM, not GK
def get_current_user(
        token: str = Depends(get_jwt),
        db: Session = Depends(get_db)
) -> User:

    user_id = decode_token(token)

    user_db = user.get(db=db, id=user_id)

    if not user_db:
        raise HTTPException(
            status_code=400,
            detail="User ID doesn't exist"
        )

    return user_db


def is_using_gatekeeper():
    if not settings.USING_GATEKEEPER:
        raise HTTPException(
            status_code=400,
            detail="Can't use this API without an instance of a gatekeeper"
        )


def is_not_using_gatekeeper():
    if settings.USING_GATEKEEPER:
        raise HTTPException(
            status_code=400,
            detail="Can't use this API while connected to a gatekeeper"
        )
