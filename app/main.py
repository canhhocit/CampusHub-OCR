import hmac
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from app.config import settings
from app.api.routes import router
from app.api.face_routes import face_router
from app.api.liveness_routes import liveness_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Dịch vụ thử nghiệm độc lập OCR nhận dạng Căn cước công dân Việt Nam (CCCD)",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

# Restrictive CORS configurable via OCR_ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PUBLIC_HEALTH_PATHS = {
    "/api/v1/health",
    "/api/v1/face/health",
    "/api/v1/liveness/health",
}

@app.middleware("http")
async def verify_internal_api_key(request: Request, call_next):
    configured_key = settings.OCR_API_KEY.strip()
    if (
        configured_key
        and request.method != "OPTIONS"
        and request.url.path.startswith("/api/v1/")
        and request.url.path not in PUBLIC_HEALTH_PATHS
    ):
        provided_key = (request.headers.get("X-Internal-Api-Key") or "").strip()
        if not provided_key or not hmac.compare_digest(provided_key, configured_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized internal service request."},
            )
    return await call_next(request)

# Disable browser caching for static files and HTML in development
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount static files & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API routers
app.include_router(router, prefix="/api/v1")
app.include_router(face_router, prefix="/api/v1/face")
app.include_router(liveness_router, prefix="/api/v1/liveness")

# Web testing interface
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "disclaimer": settings.DISCLAIMER,
        }
    )

# Generic safe error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log securely without sensitive payload details
    return JSONResponse(
        status_code=500,
        content={"detail": "Đã xảy ra sự cố nội bộ trong quá trình xử lý yêu cầu."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
