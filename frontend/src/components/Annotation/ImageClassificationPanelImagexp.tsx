// Experimental image projects — see docs/image-projects-strategy.md
import { FC, LegacyRef, useEffect, useState } from 'react';
import { CSSProperties } from 'styled-components';
import { useGetImageImagexp } from '../../core/api';
import { DisplayConfig, ElementOutModel } from '../../types';
import { AnnotationIcon } from '../Icons';

interface ImageClassificationPanelImagexpProps {
  element: ElementOutModel | undefined;
  displayConfig: DisplayConfig;
  elementId: string;
  projectSlug: string;
  frameRef: HTMLDivElement;
}

export const ImageClassificationPanelImagexp: FC<ImageClassificationPanelImagexpProps> = ({
  element,
  displayConfig,
  elementId,
  projectSlug,
  frameRef,
}) => {
  const { getImageImagexp } = useGetImageImagexp();
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!elementId || elementId === 'noelement') {
      // Keep displaying the previous image during transitions to avoid a
      // "No image" flash between annotations.
      return;
    }
    let cancelled = false;
    let nextObjectUrl: string | null = null;
    getImageImagexp(projectSlug, elementId).then((url) => {
      if (cancelled) {
        if (url) URL.revokeObjectURL(url);
        return;
      }
      nextObjectUrl = url;
      setSrc((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    });
    return () => {
      cancelled = true;
      // Do not revoke nextObjectUrl here: it is now owned by state and will
      // be revoked on the next swap or on unmount.
      void nextObjectUrl;
    };
  }, [elementId, projectSlug, getImageImagexp]);

  useEffect(() => {
    return () => {
      setSrc((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
    };
  }, []);

  return (
    <div
      className="annotation-frame"
      style={
        {
          '--height': `${displayConfig.textFrameHeight}vh`,
        } as CSSProperties
      }
      ref={frameRef as unknown as LegacyRef<HTMLDivElement>}
    >
      {element?.history && element.history[0] && element.history[0].label && (
        <span className="position-absolute end-0 top-0 me-1">
          <AnnotationIcon title={element.history[0].label} />
        </span>
      )}
      {src ? (
        <img
          src={src}
          alt={elementId}
          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
        />
      ) : (
        <p className="text-muted">No image</p>
      )}
    </div>
  );
};
