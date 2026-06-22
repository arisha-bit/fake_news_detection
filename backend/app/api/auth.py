from sqlalchemy.orm import Session

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from app.db.dependencies import get_db

from app.schemas.user import UserRegister
from app.schemas.user import UserLogin

from app.services.auth_service import create_user
from app.services.auth_service import authenticate_user

from app.core.security import create_access_token
from app.core.security import get_current_user
from app.models.user import User
from app.core.rbac import require_role


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)
@router.post("/register")
def register(
    payload: UserRegister,
    db: Session = Depends(get_db)
):

    user = create_user(
        db,
        payload.username,
        payload.email,
        payload.password
    )

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    return {
        "message": "User registered successfully"
    }
@router.post("/login")
def login(
    payload: UserLogin,
    db: Session = Depends(get_db)
):

    user = authenticate_user(
        db,
        payload.email,
        payload.password
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(
        {
            "sub": str(user.id),
            "role": user.role
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }
@router.get("/me")
def get_me(
    current_user: User = Depends(get_current_user)
):

    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role
    }
@router.get("/admin")
def admin_only(

    current_user = Depends(
        require_role("admin")
    )

):
    return {
        "message":"Welcome Admin"
    }
@router.get("/profile")
def profile(

    current_user = Depends(
        require_role(
            "user",
            "admin"
        )
    )

):
    return {
        "message":"Profile Access"
    }

