# VikataKavi portrait: how to make it

VikataKavi's face is an **original character** inspired by the era of Tenali Ramakrishna.
No portrait of the historical poet survives, so this design is new. It must **not**
resemble any actor who has played Tenali Rama, or any TV or animated version.

## Where to generate

Your Runway projects: **YTContent** has 50 credits and **NELAPATLA's project** has 0.
Generate in the Runway app (or the Developer Portal) with a photoreal image model
such as Gen-4 Image (8 credits/image) or a cheaper one (2–5 credits/image). Make
3–4 options and pick one.

**Aspect ratio:** 16:9 landscape (for example 1920×1080). Keep the face centred and
large, with head and shoulders visible.

## Main prompt

```
Photorealistic cinematic portrait of an original fictional character: a witty
16th-century Telugu court poet of the Vijayanagara empire, around 40 years old,
warm wheat-brown skin, lively intelligent eyes, a faint mischievous half-smile,
clean-shaven face with a neat thin moustache. Head shaved in front with a traditional
tuft (shikha) at the back. Vertical sandal-paste and kumkum tilak on the forehead.
Simple gold ear studs. Cream silk angavastram with a thin red-and-gold zari border
draped over one shoulder, bare-chested beneath with a rudraksha mala. Facing the
camera directly, head and shoulders centred, mouth closed, calm neutral expression
ready to speak. Soft warm key light from the front-left, gentle fill, natural skin
texture with pores. Softly blurred background of a carved stone mandapam pillar in
Hampi at golden hour. 85mm lens, shallow depth of field, ultra-detailed, natural
colours, 16:9.
```

**Things to avoid** (add as a negative prompt, or append "Avoid: ..."):
cartoon, illustration, anime, caricature, exaggerated expression, open mouth, teeth
showing, hands near face, turban covering the forehead, sunglasses, text, watermark,
multiple people, resemblance to any real actor or celebrity.

## Variations to try

- **Older sage:** "around 55 years old, a few grey strands in the moustache, laugh lines".
- **Court setting:** background "soft-focus royal court with oil lamps and silk drapes".
- **Teacher:** "holding a palm-leaf manuscript at chest height" (only if the hands stay out of the face area).

## Checks before making it a Character

1. One person only, face sharp, both eyes visible, mouth closed and relaxed.
2. No text or watermark in the image.
3. It doesn't look like any specific actor. If it does, regenerate.
4. Save the image as `web/static/vikatakavi.png`. The call page uses it as the
   poster shown before the video starts.

## Create the Runway Character

1. Go to https://dev.runwayml.com → **Characters** → **Create a Character**, in the
   project you will fund for calls.
2. Upload the chosen portrait. Name it **VikataKavi**. Personality and voice there
   don't matter: our agent supplies the Telugu voice (Sarvam) and the brain (Sarvam LLM).
3. Copy the Character / Avatar ID into the `RUNWAY_AVATAR_ID` environment variable.
