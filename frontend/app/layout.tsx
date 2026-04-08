import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StillAlive Studio",
  description: "EchoMimic V2 web studio for portrait-to-talking-video generation"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
