# clasificador-sueno @ challenger

- URI: `models:/clasificador-sueno@challenger`
- Versión: 2
- Endpoint: `POST http://127.0.0.1:5001/invocations`

## Esquema de entrada

```json
[
  {
    "type": "long",
    "name": "age",
    "required": true
  },
  {
    "type": "string",
    "name": "gender",
    "required": true
  },
  {
    "type": "string",
    "name": "chronotype",
    "required": true
  },
  {
    "type": "string",
    "name": "occupation",
    "required": true
  },
  {
    "type": "double",
    "name": "bedtime_hour",
    "required": true
  },
  {
    "type": "long",
    "name": "screen_time_before_bed_min",
    "required": true
  },
  {
    "type": "string",
    "name": "primary_app_category",
    "required": true
  },
  {
    "type": "long",
    "name": "screen_brightness_pct",
    "required": true
  },
  {
    "type": "boolean",
    "name": "blue_light_filter_on",
    "required": true
  },
  {
    "type": "long",
    "name": "caffeine_intake_mg",
    "required": true
  },
  {
    "type": "double",
    "name": "caffeine_last_hour",
    "required": true
  },
  {
    "type": "long",
    "name": "physical_activity_min",
    "required": true
  },
  {
    "type": "double",
    "name": "room_temperature_f",
    "required": true
  },
  {
    "type": "double",
    "name": "ambient_noise_db",
    "required": true
  }
]
```

## Esquema de salida

```json
[
  {
    "type": "tensor",
    "tensor-spec": {
      "dtype": "object",
      "shape": [
        -1
      ]
    }
  }
]
```

## Petición

```bash
curl http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["age", "gender", "chronotype", "occupation", "bedtime_hour", "screen_time_before_bed_min", "primary_app_category", "screen_brightness_pct", "blue_light_filter_on", "caffeine_intake_mg", "caffeine_last_hour", "physical_activity_min", "room_temperature_f", "ambient_noise_db"], "data": [[23, "Male", "Intermediate", "Unemployed", 0.18, 92, "Social Media", 81, false, 187, 12.9, 71, 79.8, 29.3]]}}'
```

En Windows PowerShell, usa `curl.exe` y guarda el cuerpo en un archivo:

```powershell
curl.exe http://127.0.0.1:5001/invocations `
  -H "Content-Type: application/json" `
  -d '@ejemplo-api/clasificador-sueno__challenger_payload.json'
```

```json
{
  "dataframe_split": {
    "columns": [
      "age",
      "gender",
      "chronotype",
      "occupation",
      "bedtime_hour",
      "screen_time_before_bed_min",
      "primary_app_category",
      "screen_brightness_pct",
      "blue_light_filter_on",
      "caffeine_intake_mg",
      "caffeine_last_hour",
      "physical_activity_min",
      "room_temperature_f",
      "ambient_noise_db"
    ],
    "data": [
      [
        23,
        "Male",
        "Intermediate",
        "Unemployed",
        0.18,
        92,
        "Social Media",
        81,
        false,
        187,
        12.9,
        71,
        79.8,
        29.3
      ]
    ]
  }
}
```

## Respuesta

```json
{
  "predictions": [
    "Mild"
  ]
}
```
