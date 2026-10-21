from app.api.routes import router
from app.api.face_routes import face_router
from app.api.liveness_routes import liveness_router

__all__ = ["router", "face_router", "liveness_router"]

