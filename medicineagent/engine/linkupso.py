# https://app.linkup.so/home

import os
from typing import Literal, Any
from datetime import date
from pydantic import BaseModel, Field

import structlog
from linkup import LinkupClient

from medicineagent import settings


logger = structlog.get_logger(__name__)
client = LinkupClient(api_key=settings.LINKUP_APIKEY)


class GenericNameSchema(BaseModel):
    medicine: list[tuple] = Field(description="List of tuples with medicine and its generic name")
    sources: list = Field(description="The list of valid and accessible sources/urls/reports to justify the information")
    

def api_search(
  query: str, 
  depth: Literal["standard", "deep"] = "standard", 
  images: bool = False, 
  response_format: Literal['searchResults', 'sourcedAnswer', 'structured'] = "structured",
  structured_output_schema: type[BaseModel] | None = None,
  from_date: date | None = None,
  to_date: date | None = None,
  exclude_domains: list[str] | None = None,
  include_domains: list[str] | None = None,
  max_results: int | None = None,
  include_inline_citations: bool | None = None,
  include_sources: bool | None = None
):
  logger.info("[LinkUp] Searching", query=query, depth=depth, images=images, response_format=response_format)
  if response_format == "structured" and structured_output_schema is None:
    structured_output_schema = GenericNameSchema 
    
  response = client.search(
    query=query,
    depth=depth,
    output_type=response_format,
    structured_output_schema=structured_output_schema,
    include_images=images,
    from_date=from_date,
    to_date=to_date,
    exclude_domains=exclude_domains,
    include_domains=include_domains,
    max_results=max_results,  
    include_inline_citations=include_inline_citations,
    include_sources=include_sources
  )
  return format_search_results(response)


def format_search_results(response: Any):
  return response

