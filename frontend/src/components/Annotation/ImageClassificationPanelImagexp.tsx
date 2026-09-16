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
    <div>
      {element?.selection === 'prompt' && element.similarity != null && (
        <small
          className="text-muted d-block mb-1"
          title="Cosine similarity between the prompt embedding and this image. Rank is the absolute position in the prompt's full ranking."
        >
          {element.rank != null && <>rank #{element.rank} · </>}
          similarity {element.similarity.toFixed(3)}
        </small>
      )}
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
          // The wrapper takes the remaining frame height and the image is
          // absolutely positioned inside it: a percent-sized <img> directly
          // in the flex column makes Safari resolve the percentage against
          // the previous layout, shrinking the image on every swap (#1127).
          <div
            style={{
              position: 'relative',
              flex: '1 1 auto',
              minHeight: 0,
              width: '100%',
              alignSelf: 'stretch',
            }}
          >
            <img
              src={src}
              alt={elementId}
              style={{
                position: 'absolute',
                inset: 0,
                margin: 'auto',
                maxWidth: '100%',
                maxHeight: '100%',
              }}
            />
          </div>
        ) : (
          <p className="text-muted">No image</p>
        )}
      </div>
    </div>
  );
};
