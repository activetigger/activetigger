// Experimental image projects — see docs/image-projects-strategy.md
// Shared thumbnail component used by Explore tabular view and the
// annotation history. Currently fetches the original image via the
// existing image_imagexp endpoint and lets the browser scale it.
// Swap the inner fetch for a real /thumbnail_imagexp/ route later
// without touching call sites.
import { CSSProperties, FC, useEffect, useState } from 'react';
import { useGetThumbnailImagexp } from '../core/api';

interface ImageThumbnailImagexpProps {
  projectSlug: string;
  elementId: string;
  maxWidth?: number;
  maxHeight?: number;
  style?: CSSProperties;
}

export const ImageThumbnailImagexp: FC<ImageThumbnailImagexpProps> = ({
  projectSlug,
  elementId,
  maxWidth = 120,
  maxHeight = 80,
  style,
}) => {
  const { getThumbnailImagexp } = useGetThumbnailImagexp();
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!projectSlug || !elementId) return;
    let cancelled = false;
    let objectUrl: string | null = null;
    getThumbnailImagexp(projectSlug, elementId).then((url) => {
      if (cancelled) {
        if (url) URL.revokeObjectURL(url);
        return;
      }
      objectUrl = url;
      setSrc(url);
    });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [projectSlug, elementId, getThumbnailImagexp]);

  if (!src) return null;
  return (
    <img
      src={src}
      alt={elementId}
      loading="lazy"
      style={{ maxWidth, maxHeight, objectFit: 'contain', ...style }}
    />
  );
};
