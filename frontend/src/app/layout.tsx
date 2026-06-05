import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Banana Bunch Detection",
  description: "YOLO-based banana bunch detection powered by CS406 AI Team",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body className="antialiased">{children}</body>
    </html>
  );
}
