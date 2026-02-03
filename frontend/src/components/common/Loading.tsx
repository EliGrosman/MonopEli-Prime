interface LoadingProps {
  message?: string;
  fullScreen?: boolean;
}

/**
 * Loading spinner component.
 */
export function Loading({ message, fullScreen = false }: LoadingProps) {
  const content = (
    <div className="flex flex-col items-center gap-4">
      <div className="w-10 h-10 border-4 border-board-border border-t-transparent rounded-full animate-spin" />
      {message && <p className="text-gray-600">{message}</p>}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-white bg-opacity-80 flex items-center justify-center z-50">
        {content}
      </div>
    );
  }

  return <div className="flex items-center justify-center p-8">{content}</div>;
}

/**
 * Loading overlay for the entire app.
 */
interface GlobalLoadingProps {
  isLoading: boolean;
  message?: string | null;
}

export function GlobalLoading({ isLoading, message }: GlobalLoadingProps) {
  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-8 shadow-xl">
        <Loading message={message ?? 'Loading...'} />
      </div>
    </div>
  );
}
