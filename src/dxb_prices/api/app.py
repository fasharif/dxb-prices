"""FastAPI application: ``POST /estimate`` plus health and metadata endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import lightgbm as lgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from dxb_prices import config
from dxb_prices.api.estimator import Estimator, UnknownCommunityError
from dxb_prices.api.schemas import EstimateRequest, EstimateResponse, Problem
from dxb_prices.model import PriceModel

log = logging.getLogger(__name__)


def create_app(model_dir: Path | None = None) -> FastAPI:
    directory = model_dir or config.MODEL_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.estimator = Estimator(PriceModel.load(directory))
            app.state.load_error = None
            log.info(
                "loaded model %s from %s",
                app.state.estimator.model.metadata.get("model_version"),
                directory,
            )
        # A missing, unreadable or corrupt model leaves the service up but unavailable.
        except (OSError, ValueError, KeyError, lgb.basic.LightGBMError) as exc:
            app.state.estimator = None
            app.state.load_error = str(exc)
            log.error("no model loaded: %s", exc)
        yield

    app = FastAPI(
        title="dxb-prices",
        version="0.1.0",
        summary="Sale price estimates for Dubai apartments, with the factors behind each one.",
        lifespan=lifespan,
    )

    def estimator(request: Request) -> Estimator:
        est: Estimator | None = request.app.state.estimator
        if est is None:
            raise HTTPException(
                status_code=503, detail=f"model not loaded: {request.app.state.load_error}"
            )
        return est

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        est: Estimator | None = request.app.state.estimator
        if est is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "detail": request.app.state.load_error},
            )
        return JSONResponse(
            {"status": "ok", "model_version": est.model.metadata.get("model_version")}
        )

    @app.get("/model")
    def model_info(request: Request) -> dict[str, Any]:
        meta = estimator(request).model.metadata
        return {
            k: meta.get(k)
            for k in (
                "model_version",
                "trained_at",
                "target",
                "trained_on_months",
                "evaluated_on_months",
                "data_period_end",
                "test_scores",
            )
        }

    @app.get("/communities")
    def communities(request: Request) -> list[dict[str, Any]]:
        rows = estimator(request).community_rows()
        return [{"name": name, "training_sales": n} for name, n in rows.items()]

    @app.post(
        "/estimate",
        response_model=EstimateResponse,
        responses={404: {"model": Problem}, 503: {"model": Problem}},
    )
    def estimate(body: EstimateRequest, request: Request) -> EstimateResponse | JSONResponse:
        est = estimator(request)
        try:
            return est.estimate(body)
        except UnknownCommunityError as exc:
            return JSONResponse(
                status_code=404,
                content=Problem(
                    detail=f"Community '{exc.name}' is not in the training data.",
                    suggestions=exc.suggestions,
                ).model_dump(),
            )

    return app


app = create_app()
