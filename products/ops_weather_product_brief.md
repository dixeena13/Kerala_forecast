# Operational Weather Product Brief

This is a simulated product-design brief for portfolio purposes, not an operational MeteoSwiss or internal operational product.
It demonstrates how user requirements can be translated into weather-product concepts and update frequencies.

| Mission profile | User group | Operational question | Recommended weather products | Update frequency |
|---|---|---|---|---|
| Air operations | aviation coordination | Can aircraft or drones operate safely in the next 6-24 hours? | wind gust nowcast; visibility forecast; thunderstorm alert; freezing-level forecast | hourly / event-driven |
| Ground mobility | field logistics | Where can rain, snow, or poor visibility disrupt movement and supply routes? | precipitation accumulation map; road-weather risk layer; visibility forecast | 3-hourly |
| Mountain support | alpine operations | Which weather hazards affect exposed terrain, freezing level, and local accessibility? | freezing-level forecast; wind gust forecast; convective risk briefing; snow/rain transition layer | 3-hourly / briefing before deployment |
| Civil protection support | coordination staff | Which weather developments require early coordination with civil partners? | impact-based warning dashboard; heavy precipitation scenario; thunderstorm alert; heat stress outlook | daily / event-driven |

## Product Management Notes

- Start with mission-specific questions rather than a generic weather dashboard.
- Translate each question into hazards, thresholds, product formats, and update frequency.
- Keep traceability from user requirement to meteorological variable and IT product component.
- Review products with users after exercises or high-impact weather events.
